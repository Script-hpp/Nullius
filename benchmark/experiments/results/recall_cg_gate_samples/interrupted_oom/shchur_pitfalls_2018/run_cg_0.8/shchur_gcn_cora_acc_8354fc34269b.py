

# Plan step -> Load data from /data/cora_lcc_public.npz via nullius_io
# Plan step -> Preprocess adjacency: add self-loops, symmetrically normalize, convert to sparse tensor
# Plan step -> Prepare features as sparse tensor (no normalization)
# Plan step -> Define GCN model with configurable number of layers as per CONFIG["layers"]
# Plan step -> Set up Adam optimizer with learning rate from CONFIG, manual L2 weight decay on first layer
# Plan step -> Training loop over epochs up to CONFIG["epochs"], compute training loss on train_idx, validation loss on val_idx, apply early stopping (gnnbench criterion, patience CONFIG["patience"])
# Plan step -> After training, restore best model, compute logits for all nodes, predict class via argmax
# Plan step -> Write predictions via nullius_io.write_predictions and print validation accuracy

import os
import numpy as np
import torch
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {
    "lr": 0.01,
    "epochs": 100000,
    "layers": 2,
    "patience": 50,
    "weight_decay": 0.001,
    "hidden": 64
}

def main():
    # seed
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load data
    d = nio.load()
    features = torch.from_numpy(d["features"])  # [N, F] float32
    edges = torch.from_numpy(d["edges"])        # [E, 2] int64
    labels = torch.from_numpy(d["labels"])      # [N] int64, -1 for test
    train_idx = torch.from_numpy(d["train_idx"]).long()
    val_idx = torch.from_numpy(d["val_idx"]).long()
    test_idx = torch.from_numpy(d["test_idx"]).long()

    N = features.shape[0]
    F = features.shape[1]

    # --- adjacency preprocessing ---
    # add self-loops
    self_loop = torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(1, 2)
    self_loop[:, 1] = torch.arange(N, dtype=torch.long)
    edge_index = torch.cat([edges, self_loop], dim=0)  # [E+ N, 2]
    # coalesce and get values (all ones)
    adj = torch.sparse_coo_tensor(edge_index.t(), torch.ones(edge_index.size(0), dtype=torch.float32), (N, N))
    adj = adj.coalesce()

    # compute degree
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # [N]
    deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # normalize: D^{-1/2} A D^{-1/2}
    indices = adj.indices()
    vals = adj.values()
    cols = indices[1, :]
    vals = vals * deg_inv_sqrt[cols]
    rows = indices[0, :]
    vals = vals * deg_inv_sqrt[rows]
    adj_norm = torch.sparse_coo_tensor(indices, vals, (N, N)).coalesce()

    # --- features as sparse tensor (no normalization) ---
    features_nonzero = features.nonzero(as_tuple=False).t()
    features_vals = features[features_nonzero[0], features_nonzero[1]]
    features_sparse = torch.sparse_coo_tensor(features_nonzero, features_vals, (N, F)).coalesce()

    # --- GCN model ---
    class GCN(torch.nn.Module):
        def __init__(self, in_features, hidden_features, out_features, num_layers, dropout, weight_decay):
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            self.weight_decay = weight_decay
            self.layers = torch.nn.ModuleList()
            # first layer
            self.layers.append(torch.nn.Linear(in_features, hidden_features, bias=True))
            # hidden layers (if any)
            for _ in range(num_layers - 2):
                self.layers.append(torch.nn.Linear(hidden_features, hidden_features, bias=True))
            # last layer
            self.layers.append(torch.nn.Linear(hidden_features, out_features, bias=True))
            # initialize
            for lin in self.layers:
                torch.nn.init.glorot_uniform_(lin.weight)
                torch.nn.init.zeros_(lin.bias)

        def _dropout_sparse(self, x):
            """Inverted dropout for sparse tensor"""
            if not x.is_coalesced():
                x = x.coalesce()
            vals = x.values()
            mask = torch.rand_like(vals) > self.dropout  # keep probability = 1 - dropout
            vals = vals * mask.float() / (1.0 - self.dropout)
            return torch.sparse_coo_tensor(x.indices(), vals, x.size())

        def forward(self, adj_norm, features_sparse):
            x = features_sparse
            for i, lin in enumerate(self.layers):
                # dropout on inputs (if training)
                if self.training:
                    if isinstance(x, torch.sparse.Tensor):
                        x = self._dropout_sparse(x)
                    else:
                        x = F.dropout(x, p=self.dropout, training=True)
                # support = x * W
                if isinstance(x, torch.sparse.Tensor):
                    support = torch.sparse.mm(x, lin.weight)
                else:
                    support = torch.mm(x, lin.weight)
                # graph convolution: adj_norm * support
                out = torch.sparse.mm(adj_norm, support) + lin.bias
                if i < len(self.layers) - 1:  # not last layer
                    out = F.relu(out)
                x = out
            return x

        def l2_loss(self):
            # L2 penalty only on first layer's weight matrix
            return self.weight_decay * torch.norm(self.layers[0].weight, p=2) ** 2

    model = GCN(
        in_features=F,
        hidden_features=CONFIG["hidden"],
        out_features=7,
        num_layers=CONFIG["layers"],
        dropout=0.8,
        weight_decay=CONFIG["weight_decay"]
    )
    optimizer = torch.optim.Adam(list(model.parameters()), lr=CONFIG["lr"], weight_decay=0.0)

    # training loop
    epochs = 1 if os.environ.get("SMOKE") == "1" else CONFIG["epochs"]
    patience = CONFIG["patience"]
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(adj_norm, features_sparse)
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        l2 = model.l2_loss()
        loss = loss_train + l2
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            logits_eval = model(adj_norm, features_sparse)
            loss_val = F.cross_entropy(logits_eval[val_idx], labels[val_idx])
            acc_val = (logits_eval[val_idx].argmax(dim=1) == labels[val_idx]).float().mean().item()
        model.train()

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_all = model(adj_norm, features_sparse)
        preds_all = logits_all.argmax(dim=1).cpu().numpy()
        acc_val_best = (logits_all[val_idx].argmax(dim=1) == labels[val_idx]).float().mean().item()

    print(f"Validation accuracy: {acc_val_best:.4f}")

    # write predictions (must be for all nodes)
    nio.write_predictions(preds_all)

if __name__ == "__main__":
    main()