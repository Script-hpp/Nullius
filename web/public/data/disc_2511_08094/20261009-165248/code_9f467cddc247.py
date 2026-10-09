

# Plan step -> implemented in: imports and CONFIG
# Plan step -> implemented in: seeding
# Plan step -> implemented in: data loading
# Plan step -> implemented in: graph preprocessing (adjacency, degree, normalization)
# Plan step -> implemented in: SLGCN model definition
# Plan step -> implemented in: optimizer, scheduler, training loop
# Plan step -> implemented in: smoke mode handling
# Plan step -> implemented in: prediction and output
# Plan step -> implemented in: printing validation accuracy

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CONFIG = {"lr": 0.001, "weight_decay": 1e-05, "epochs": 100, "batch_size": 64, "dropout": 0.0, "hidden": 128, "layers": 10, "patience": 100, "heads": 2}
# Additional SL-GCN specific hyperparameters
CONFIG.update({
    "zeta_real": 0.0,
    "zeta_imag": 1.0,
    "nu_real": 1.0,
    "nu_imag": 1.0,
    "dt": 1.0,
    "k": 0.3,
    "p": 2.0,
    "tol": 1e-3,
    "leaky_relu_slope": 0.3,
    "bias": False,
    "input_dropout": 0.0,
})

def main():
    # Seeding
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Smoke mode handling
    smoke = os.environ.get("SMOKE") == "1"
    max_epochs = 1 if smoke else CONFIG["epochs"]

    # Data loading
    import nullius_io as nio
    data = nio.load()
    features = data["features"]          # np.ndarray [N, 1433] float32
    edges = data["edges"]                # np.ndarray [E, 2] int64
    labels = data["labels"]              # np.ndarray [N] int64, -1 for test
    train_idx = data["train_idx"]        # np.ndarray [140]
    val_idx = data["val_idx"]            # np.ndarray [210]
    test_idx = data["test_idx"]          # np.ndarray [2135]

    N = features.shape[0]
    feat_dim = features.shape[1]
    num_classes = 7

    # Torch tensors
    features_t = torch.from_numpy(features).float()
    labels_t = torch.from_numpy(labels).long()

    # Masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Graph preprocessing (adjacency, degree, normalization)
    edges_t = torch.from_numpy(edges).long()
    E = edges_t.shape[0]
    adj_sp = torch.sparse_coo_tensor(
        edges_t.t(),
        torch.ones(E, dtype=torch.float32),
        size=(N, N),
    ).coalesce()

    deg = torch.bincount(edges_t[:, 0], minlength=N).float()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # SL-GCN specific parameters
    zeta = complex(CONFIG["zeta_real"], CONFIG["zeta_imag"])
    nu = complex(CONFIG["nu_real"], CONFIG["nu_imag"])
    factor = CONFIG["dt"] * (zeta + nu)

    # SLGCN model definition
    class SLGCN(nn.Module):
        def __init__(self, feat_dim, hidden, num_classes, num_layers,
                     dropout, input_dropout, leaky_relu_slope, bias,
                     heads, factor, deg_inv_sqrt, adj_sp):
            super().__init__()
            self.num_layers = num_layers
            self.hidden = hidden
            self.dropout = dropout
            self.input_dropout = input_dropout
            self.leaky_relu_slope = leaky_relu_slope
            self.bias = bias
            self.heads = heads
            self.factor = factor
            self.register_buffer('deg_inv_sqrt', deg_inv_sqrt)
            self.register_buffer('adj_sp', adj_sp)

            self.linears = nn.ModuleList()
            # Input layer: feat_dim -> hidden * heads
            self.linears.append(nn.Linear(feat_dim, hidden * heads, bias=bias))
            # Hidden layers: hidden * heads -> hidden * heads (for each of the next num_layers-2)
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden * heads, hidden * heads, bias=bias))
            # Output layer: hidden * heads -> num_classes
            if num_layers > 1:
                self.linears.append(nn.Linear(hidden * heads, num_classes, bias=bias))
            else:
                self.linears.append(nn.Linear(feat_dim, num_classes, bias=bias))

            for lin in self.linears:
                nn.init.xavier_uniform_(lin.weight)
                if bias:
                    nn.init.zeros_(lin.bias)

        def _propagate(self, h):
            # h: complex [N, D]
            # Compute Lh = h - D^{-1/2} A (D^{-1/2} h)
            dh_real = self.deg_inv_sqrt.view(-1, 1) * h.real
            dh_imag = self.deg_inv_sqrt.view(-1, 1) * h.imag
            Ah_real = torch.sparse.mm(self.adj_sp, dh_real)
            Ah_imag = torch.sparse.mm(self.adj_sp, dh_imag)
            Ah = torch.complex(Ah_real, Ah_imag)
            dh2_real = self.deg_inv_sqrt.view(-1, 1) * Ah.real
            dh2_imag = self.deg_inv_sqrt.view(-1, 1) * Ah.imag
            dh2 = torch.complex(dh2_real, dh2_imag)
            Lh = h - dh2
            return h + self.factor * Lh

        def forward(self, x):
            if self.training and self.input_dropout > 0:
                x = F.dropout(x, p=self.input_dropout, training=True)
            h = torch.complex(x, torch.zeros_like(x))
            for i, lin in enumerate(self.linears):
                h = self._propagate(h)
                h_real = h.real
                h_imag = h.imag
                h_real = lin(h_real)
                h_imag = lin(h_imag)
                if i < self.num_layers - 1:
                    # Reshape for multi-head combination: [N, heads, hidden]
                    out_dim = self.hidden * self.heads
                    h_real = h_real.view(-1, self.heads, self.hidden)
                    h_imag = h_imag.view(-1, self.heads, self.hidden)
                    # Combine heads via mean
                    h_real = h_real.mean(dim=1)
                    h_imag = h_imag.mean(dim=1)
                    # Dropout
                    if self.dropout > 0:
                        h_real = F.dropout(h_real, p=self.dropout, training=self.training)
                        h_imag = F.dropout(h_imag, p=self.dropout, training=self.training)
                    # Activation
                    h_real = F.leaky_relu(h_real, negative_slope=self.leaky_relu_slope)
                    h_imag = F.leaky_relu(h_imag, negative_slope=self.leaky_relu_slope)
                    h = torch.complex(h_real, h_imag)
                else:
                    # Last layer: output logits (real part)
                    h = torch.complex(h_real, h_imag)
            return h.real

    model = SLGCN(
        feat_dim=feat_dim,
        hidden=CONFIG["hidden"],
        num_classes=num_classes,
        num_layers=CONFIG["layers"],
        dropout=CONFIG["dropout"],
        input_dropout=CONFIG["input_dropout"],
        leaky_relu_slope=CONFIG["leaky_relu_slope"],
        bias=CONFIG["bias"],
        heads=CONFIG["heads"],
        factor=factor,
        deg_inv_sqrt=deg_inv_sqrt,
        adj_sp=adj_sp,
    )

    # Optimizer, scheduler, training loop
    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=CONFIG["patience"]
    )
    lr_initial = optimizer.param_groups[0]["lr"]
    lr_stop = lr_initial * 0.9

    best_val_acc = 0.0
    best_state = None

    loader = DataLoader(train_idx.tolist(), batch_size=CONFIG["batch_size"], shuffle=True)

    model.train()
    for epoch in range(max_epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(features_t)  # full forward pass
            loss = F.cross_entropy(out[batch], labels_t[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        model.eval()
        with torch.no_grad():
            out = model(features_t)
            pred = out.argmax(dim=1)
            def acc(mask):
                correct = (pred[mask] == labels_t[mask]).sum().item()
                total = mask.sum().item()
                return correct / total * 100.0 if total > 0 else 0.0
            train_acc = acc(train_mask)
            val_acc = acc(val_mask)
            test_acc = acc(test_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < lr_stop:
            break
        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Prediction and output
    model.eval()
    with torch.no_grad():
        logits = model(features_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    nio.write_predictions(preds)

    # Printing validation accuracy
    print(f"{best_val_acc:.2f}")

if __name__ == "__main__":
    main()