

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

CONFIG = {
    "lr": 0.001,
    "weight_decay": 1e-05,
    "epochs": 100,
    "batch_size": 64,
    "dropout": 0.0,
    "hidden": 128,
    "layers": 10,
    "patience": 100,
    "heads": 2,
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
}

def main():
    # Seed
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Smoke mode
    smoke = os.environ.get("SMOKE") == "1"
    max_epochs = 1 if smoke else CONFIG["epochs"]

    # Load data
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

    # Build sparse adjacency (undirected, both directions present)
    edges_t = torch.from_numpy(edges).long()
    E = edges_t.shape[0]
    adj_sp = torch.sparse_coo_tensor(
        edges_t.t(),
        torch.ones(E, dtype=torch.float32),
        size=(N, N),
    ).coalesce()

    # Degree (using source nodes; each undirected edge appears once as source per endpoint)
    deg = torch.bincount(edges_t[:, 0], minlength=N).float()
    # Avoid division by zero
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    deg_inv_sqrt = deg_inv_sqrt  # will be registered as buffer

    # Propagation factor (complex scalar)
    zeta = complex(CONFIG["zeta_real"], CONFIG["zeta_imag"])
    nu = complex(CONFIG["nu_real"], CONFIG["nu_imag"])
    factor = CONFIG["dt"] * (zeta + nu)  # complex

    # SLGCN model
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
            assert hidden % heads == 0, "hidden must be divisible by heads"
            self.head_dim = hidden // heads
            self.factor = factor  # complex scalar
            self.register_buffer('deg_inv_sqrt', deg_inv_sqrt)
            self.register_buffer('adj_sp', adj_sp)

            self.linears = nn.ModuleList()
            # Input layer
            self.linears.append(nn.Linear(feat_dim, hidden * heads, bias=bias))
            # Hidden layers
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden, hidden * heads, bias=bias))
            # Output layer (if num_layers > 1)
            if num_layers > 1:
                self.linears.append(nn.Linear(hidden, num_classes, bias=bias))
            else:
                # single layer case: map directly from feat_dim to classes
                self.linears.append(nn.Linear(feat_dim, num_classes, bias=bias))

            # Init weights
            for lin in self.linears:
                nn.init.xavier_uniform_(lin.weight)
                if bias:
                    nn.init.zeros_(lin.bias)

        def _propagate(self, h):
            # h: [N, F] complex
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
            # h + factor * Lh
            return h + self.factor * Lh

        def forward(self, x):
            # Input dropout
            if self.training and self.input_dropout > 0:
                x = F.dropout(x, p=self.input_dropout, training=True)
            # Start with real features, zero imag
            h = torch.complex(x, torch.zeros_like(x))
            for i, lin in enumerate(self.linears):
                h = self._propagate(h)
                h_real = h.real
                h_imag = h.imag
                # Linear weight applied separately to real and imag
                h_real = lin(h_real)
                h_imag = lin(h_imag)
                if i < self.num_layers - 1:
                    # Reshape for multi-head combination
                    out_dim = hidden * self.heads
                    h_real = h_real.view(-1, self.heads, self.head_dim)
                    h_imag = h_imag.view(-1, self.heads, self.head_dim)
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
            # Return real part as logits
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
    best_test_acc = 0.0
    best_state = None

    # Prepare DataLoader for mini-batch training over train indices
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
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < lr_stop:
            break

        model.train()

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final prediction for all nodes
    model.eval()
    with torch.no_grad():
        logits = model(features_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    # Write predictions via helper
    nio.write_predictions(preds)

    # Print validation accuracy
    print(f"{best_val_acc:.2f}")

if __name__ == "__main__":
    main()