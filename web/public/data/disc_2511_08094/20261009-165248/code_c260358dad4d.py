

# Plan step -> implemented in: imports
# Plan step -> implemented in: seeding
# Plan step -> implemented in: data loading
# Plan step -> implemented in: graph preprocessing
# Plan step -> implemented in: SLGCN model definition
# Plan step -> implemented in: optimizer, scheduler, training loop
# Plan step -> implemented in: prediction and output
# Plan step -> implemented in: printing validation accuracy

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CONFIG = {"lr": 0.001, "weight_decay": 1e-05, "epochs": 100, "batch_size": 64, "dropout": 0.0, "hidden": 128, "layers": 10, "patience": 100, "heads": 2}
# Additional SL‑GCN specific hyperparameters
CONFIG.update(
    {
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
)


def main():
    # ---------- seeding ----------
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------- data loading ----------
    import nullius_io as nio

    data = nio.load()
    features = data["features"]  # np.ndarray [N, 1433] float32
    edges = data["edges"]  # np.ndarray [E, 2] int64
    labels = data["labels"]  # np.ndarray [N] int64, -1 for test
    train_idx = data["train_idx"]  # np.ndarray [140]
    val_idx = data["val_idx"]  # np.ndarray [210]
    test_idx = data["test_idx"]  # np.ndarray [2135]

    N = features.shape[0]
    feat_dim = features.shape[1]
    num_classes = 7

    # torch tensors
    features_t = torch.from_numpy(features).float()
    labels_t = torch.from_numpy(labels).long()

    # masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # ---------- graph preprocessing (adjacency, degree, normalization) ----------
    edges_t = torch.from_numpy(edges).long()  # [E, 2]

    # add self-loops
    self_loop = torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(1, 2)  # [N, 2]
    edge_index = torch.cat([edges_t, self_loop], dim=0)  # [E+N, 2]

    # compute degree and inverse sqrt
    deg = torch.bincount(edge_index[:, 0], minlength=N).float()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # sparse adjacency matrix (with self-loops)
    adj_sp = torch.sparse_coo_tensor(
        edge_index.t(),
        torch.ones(edge_index.size(0), dtype=torch.float32),
        size=(N, N),
    ).coalesce()

    # ---------- SLGCN model definition ----------
    zeta = complex(CONFIG["zeta_real"], CONFIG["zeta_imag"])
    nu = complex(CONFIG["nu_real"], CONFIG["nu_imag"])
    factor = CONFIG["dt"] * (zeta + nu)  # complex scalar

    class ComplexLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight_re = nn.Parameter(torch.Tensor(out_features, in_features))
            self.weight_im = nn.Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.bias_re = nn.Parameter(torch.Tensor(out_features))
                self.bias_im = nn.Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias_re', None)
                self.register_parameter('bias_im', None)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight_re)
            nn.init.xavier_uniform_(self.weight_im)
            if self.bias_re is not None:
                nn.init.zeros_(self.bias_re)
                nn.init.zeros_(self.bias_im)

        def forward(self, input):
            # input: complex tensor [N, in_features]
            real = input.real
            imag = input.imag
            # output_real = real @ W_re^T - imag @ W_im^T
            # output_imag = real @ W_im^T + imag @ W_re^T
            output_real = F.linear(real, self.weight_re, self.bias_re) - F.linear(imag, self.weight_im, self.bias_im)
            output_imag = F.linear(real, self.weight_im, self.bias_im) + F.linear(imag, self.weight_re, self.bias_re)
            return torch.complex(output_real, output_imag)

    class SLGCN(nn.Module):
        def __init__(
            self,
            feat_dim,
            hidden,
            num_classes,
            num_layers,
            dropout,
            input_dropout,
            leaky_relu_slope,
            bias,
            heads,
            factor,
            deg_inv_sqrt,
            adj_sp,
        ):
            super().__init__()
            self.num_layers = num_layers
            self.hidden = hidden
            self.dropout = dropout
            self.input_dropout = input_dropout
            self.leaky_relu_slope = leaky_relu_slope
            self.bias = bias
            self.heads = heads
            self.factor = factor
            self.register_buffer("deg_inv_sqrt", deg_inv_sqrt)
            self.register_buffer("adj_sp", adj_sp)

            self.linears = nn.ModuleList()
            # input layer
            self.linears.append(ComplexLinear(feat_dim, hidden * heads, bias=bias))
            # hidden layers
            for _ in range(num_layers - 2):
                self.linears.append(ComplexLinear(hidden * heads, hidden * heads, bias=bias))
            # output layer
            if num_layers > 1:
                self.linears.append(ComplexLinear(hidden * heads, num_classes, bias=bias))
            else:
                self.linears.append(ComplexLinear(feat_dim, num_classes, bias=bias))

        def _propagate(self, h):
            # h: complex [N, D]
            # Lh = h - D^{-1/2} A (D^{-1/2} h)
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
            h = self.linears[0](x)  # [N, hidden*heads]
            h = torch.complex(h, torch.zeros_like(h))
            for i in range(self.num_layers - 1):
                h = self._propagate(h)
                h = self.linears[i + 1](h)
                if i < self.num_layers - 2:
                    if self.dropout > 0:
                        h_real = F.dropout(h.real, p=self.dropout, training=self.training)
                        h_imag = F.dropout(h.imag, p=self.dropout, training=self.training)
                    else:
                        h_real = h.real
                        h_imag = h.imag
                    h_real = F.leaky_relu(
                        h_real, negative_slope=self.leaky_relu_slope
                    )
                    h_imag = F.leaky_relu(
                        h_imag, negative_slope=self.leaky_relu_slope
                    )
                    h = torch.complex(h_real, h_imag)
            return h.real  # [N, num_classes]

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

    # ---------- optimizer, scheduler, training loop ----------
    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=CONFIG["patience"]
    )
    lr_initial = optimizer.param_groups[0]["lr"]
    lr_stop = lr_initial * 0.9

    best_val_acc = 0.0
    best_state = None

    # Use full-batch for simplicity (still respects batch_size conceptually)
    model.train()
    for epoch in range(CONFIG["epochs"]):
        optimizer.zero_grad()
        out = model(features_t)  # [N, num_classes]
        loss = F.cross_entropy(out[train_idx], labels_t[train_idx])
        loss.backward()
        optimizer.step()

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
            # test_acc = acc(test_mask)  # not used for early stopping

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

    # ---------- prediction and output ----------
    model.eval()
    with torch.no_grad():
        logits = model(features_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    nio.write_predictions(preds)

    # ---------- printing validation accuracy ----------
    print(f"{best_val_acc:.2f}")


if __name__ == "__main__":
    main()