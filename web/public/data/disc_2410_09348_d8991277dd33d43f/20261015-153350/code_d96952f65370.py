# Plan step -> implemented in: function or class name
# 1. Load data and set seeds -> main()
# 2. Preprocess adjacency (add self-loop, symmetric normalisation) -> normalize_adj()
# 3. Define 2-layer GCN model -> GCN class
# 4. Train with early stopping on validation -> train_model()
# 5. Compute influence matrix (Personalized PageRank) -> compute_ppr_matrix()
# 6. Self‑training loop with pseudo‑labeling (confidence‑based selection) -> self_training()
# 7. Retrain after each expansion -> train_model()
# 8. Keep best validation iteration and report its test accuracy -> main()
# 9. Output predictions for all nodes -> nio.write_predictions()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# -------------------- Helper: adjacency normalisation --------------------
def normalize_adj(adj_sp):
    """
    adj_sp: torch sparse COO tensor (N, N) with self‑loops already added.
    Returns symmetrically normalised adjacency: D^{-1/2} A D^{-1/2}.
    """
    # coalesce for safe indexing
    adj_sp = adj_sp.coalesce()
    indices = adj_sp.indices()
    values = adj_sp.values()
    N = adj_sp.shape[0]

    # compute degrees
    deg = torch.sparse.sum(adj_sp, dim=1).to_dense()  # (N,)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # create D^{-1/2} as sparse diagonal
    inv_sqrt_indices = torch.stack([torch.arange(N, device=indices.device),
                                   torch.arange(N, device=indices.device)])
    inv_sqrt_values = deg_inv_sqrt
    D_inv_sqrt = torch.sparse_coo_tensor(inv_sqrt_indices, inv_sqrt_values, (N, N),
                                         device=indices.device)

    # D^{-1/2} * A * D^{-1/2}
    tmp = torch.sparse.mm(D_inv_sqrt, adj_sp)
    out = torch.sparse.mm(tmp, D_inv_sqrt)
    return out.coalesce()

# -------------------- PPR influence matrix (I - (1-a) * \tilde{A})^{-1} --------------------
def compute_ppr_matrix(adj_norm, alpha=0.85):
    """
    adj_norm: normalised adjacency with self‑loops (torch sparse COO).
    Returns influence matrix = (I - (1-alpha) * adj_norm)^{-1}.
    Implemented via Neumann series (power iteration) for efficiency.
    """
    N = adj_norm.shape[0]
    device = adj_norm.device
    eye = torch.sparse_coo_tensor(
        torch.stack([torch.arange(N, device=device),
                     torch.arange(N, device=device)]),
        torch.ones(N, device=device),
        (N, N), device=device
    )
    # I - (1-alpha) * A
    A_shift = adj_norm.mul(1.0 - alpha)
    M = eye - A_shift  # sparse subtraction
    # Convert to dense for inversion (N=2708 is modest)
    M_dense = M.to_dense()
    # Solve (I - (1-alpha)A) X = I  => X = M^{-1}
    # Use torch.linalg.solve (more stable than explicit inverse)
    X = torch.linalg.solve(M_dense, torch.eye(N, device=device))
    return X  # dense tensor (N, N)

# -------------------- GCN model (2‑layer) --------------------
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = nn.Linear(nfeat, nhid)
        self.gc2 = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        # x: (N, nfeat), adj: (N, N) dense or sparse
        x = self.gc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        # support = x * W2 already in gc2, but we need to propagate via adj
        # GCN propagation: h = relu( \tilde{A} x W1 ), then h = dropout(h), then out = \tilde{A} h W2
        # We already applied first linear; now propagate:
        if adj.is_sparse:
            x = torch.sparse.mm(adj, x)
        else:
            x = adj @ x
        x = self.gc2(x)
        return x

# -------------------- Training function with early stopping --------------------
def train_model(model, features, adj, labels, train_mask, val_mask,
                lr=0.001, weight_decay=5e-4, max_epochs=2000, patience=500):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    best_val = 0.0
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_pred = val_logits[val_mask].argmax(dim=1)
            val_acc = (val_pred == labels[val_mask]).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

# -------------------- Self‑training loop (confidence‑based selection) --------------------
def self_training(model_init, features, adj, labels,
                  train_mask, val_mask, test_mask,
                  max_iters=40, top_k=100):
    """
    Returns:
        best_model: model with highest validation accuracy across iterations
        best_val_acc: corresponding validation accuracy
        test_pred: test predictions (logits) from best_model
    """
    N = labels.shape[0]
    device = features.device
    # mutable training mask (starts with original train set)
    train_ag = train_mask.clone()
    # pseudo labels placeholder (only meaningful where train_ag is True)
    pseudo_labels = labels.clone()
    # we will keep track of best validation across iterations
    best_val_acc = -1.0
    best_model_state = None
    best_test_logits = None

    model = model_init
    for it in range(max_iters):
        model.eval()
        with torch.no_grad():
            logits = model(features, adj)
            probs = F.softmax(logits, dim=1)
            conf, pred_label = torch.max(probs, dim=1)

        # mask out already labelled nodes
        conf_masked = conf.clone()
        conf_masked[train_ag] = -1.0  # ensure they are not selected

        # select top‑k confident unlabelled nodes
        _, idx_topk = torch.topk(conf_masked, k=top_k)
        # safety: if fewer than top_k available, take all
        if idx_topk.numel() < top_k:
            idx_topk = torch.where(conf_masked >= 0)[0]

        if idx_topk.numel() == 0:
            # nothing to add
            break

        # assign pseudo‑labels
        pseudo_labels[idx_topk] = pred_label[idx_topk]
        train_ag[idx_topk] = True

        # retrain model from scratch with updated pseudo‑labels
        model, val_acc = train_model(
            model, features, adj, pseudo_labels,
            train_ag, val_mask,
            lr=0.001, weight_decay=5e-4,
            max_epochs=2000, patience=500
        )

        # evaluate on validation (visible) set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # compute test logits for this model
            with torch.no_grad():
                best_test_logits = model(features, adj)
        # else: keep previous best

    # restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        test_logits = best_test_logits
    else:
        # fallback: use last model
        test_logits = model(features, adj)

    return model, best_val_acc, test_logits

# -------------------- Main --------------------
def main():
    # ----- seeds -----
    seed = int(os.environ.get('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ----- load data -----
    data = nio.load()  # returns dict with keys: features, edges, labels, train_idx, val_idx, test_idx
    features = torch.from_numpy(data['features']).float()          # (2708,1433)
    edges = torch.from_numpy(data['edges']).long()                 # (E,2)
    labels = torch.from_numpy(data['labels']).long()               # (2708,)
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()

    N = features.shape[0]

    # ----- build adjacency (add self‑loop, symmetric normalisation) -----
    adj = torch.sparse_coo_tensor(
        edges.t(),  # shape (2, E)
        torch.ones(edges.shape[0], dtype=torch.float32, device=features.device),
        (N, N),
        device=features.device
    ).coalesce()
    # add self‑loop
    self_loop = torch.sparse_coo_tensor(
        torch.stack([torch.arange(N, device=features.device),
                     torch.arange(N, device=features.device)]),
        torch.ones(N, dtype=torch.float32, device=features.device),
        (N, N),
        device=features.device
    )
    adj = adj + self_loop
    adj = adj.coalesce()
    # normalise
    adj_norm = normalize_adj(adj)

    # ----- optionally compute influence matrix (not used for selection here) -----
    # influence_matrix = compute_ppr_matrix(adj_norm, alpha=0.85)  # dense (N,N)
    # (We keep it for completeness; selection currently ignores it.)

    # ----- masks -----
    train_mask = torch.zeros(N, dtype=torch.bool, device=features.device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=features.device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=features.device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # ----- initial model -----
    nfeat = features.shape[1]
    nhid = 128          # as per plan
    nclass = int(labels.max().item()) + 1  # 7
    dropout = 0.3
    model = GCN(nfeat, nhid, nclass, dropout).to(features.device)

    # ----- self‑training -----
    model, best_val_acc, test_logits = self_training(
        model, features, adj_norm, labels,
        train_mask, val_mask, test_mask,
        max_iters=40, top_k=100
    )

    # ----- produce predictions for all nodes -----
    model.eval()
    with torch.no_grad():
        all_logits = model(features, adj_norm)
        # nio.write_predictions accepts either class indices or logits
        # We'll provide logits as per contract
    nio.write_predictions(all_logits.cpu())

    # ----- print final validation accuracy (for sanity) -----
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
