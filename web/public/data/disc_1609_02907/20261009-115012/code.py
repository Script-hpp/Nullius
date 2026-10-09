# Plan step -> implemented in: function or class name
# Model: two‑layer GCN (input → hidden = 16 → output = 7 classes); ReLU after first layer, linear after second. -> GCN class
# Features: row‑normalized (each node’s feature vector divided by its sum) -> preprocess_features
# Adjacency: renormalization trick Â = A + I, ĤÂ = Ď⁻¹ᐟ² Â Ď⁻¹ᐟ² -> build_adjacency
# Weight initialization: Glorot (Xavier) uniform -> weight_init
# Optimizer: Adam, learning_rate = 0.01; weight_decay (L2) = 5·10⁻⁴ applied to the first GCN layer weights -> configure_optimizer
# Dropout: rate = 0.5 applied to hidden‑layer inputs (sparse dropout on features and hidden activations) -> dropout layers in forward
# Epochs: maximum 200, early stopping with patience = 10 (stop if validation loss does not improve for 10 consecutive epochs) -> training loop with early stopping
# Validation set: 500 nodes (val_idx) used only for early‑stopping; not used for training gradients -> evaluate function uses val_idx
# Training loss: computed only on train_idx (140 nodes); validation and test labels are masked out -> loss computed with train_mask
# Batch size: full‑batch (all nodes processed each epoch) -> forward pass processes all nodes
# Number of layers: 2; no attention heads -> GCN class has two GraphConvolution layers
# Random seed: read from environment variable SEED, set NumPy and PyTorch seeds -> set_seeds
# After training, restore the model with lowest validation loss (or last epoch if no early stop) and compute logits for test_idx. -> train function returns best state
# Output: CSV `/output/predictions.csv` with header `id,prediction`; `id` = node index from test_idx, `prediction` = argmax class. -> nio.write_predictions call
# Smoke mode: if SMOKE=1 train exactly 1 epoch. -> condition in main

import os
import numpy as np
import torch
import torch.nn.functional as F
import nullius_io as nio

# Hyperparameters from spec
CONFIG = {"lr": 0.01, "epochs": 200, "dropout": 0.5, "hidden": 16, "patience": 10}
WEIGHT_DECAY = 5e-4  # from paper, applied to first GCN layer


def set_seeds():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_features(feats: np.ndarray) -> torch.Tensor:
    """Row‑normalize feature matrix (each node divided by its sum)."""
    row_sum = feats.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1  # avoid division by zero
    feats_norm = feats / row_sum
    return torch.from_numpy(feats_norm).float()


def build_adjacency(edges: np.ndarray, num_nodes: int) -> torch.Tensor:
    """Adjacency matrix with self‑loops and symmetric normalization (Â = D^{-1/2}(A+I)D^{-1/2})."""
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    # add self‑loops
    adj += torch.eye(num_nodes, dtype=torch.float32)
    # degree
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # normalize
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    return adj_norm


def weight_init(tensor):
    """Glorot (Xavier) uniform initialization."""
    torch.nn.init.xavier_uniform_(tensor)


class GraphConvolution(torch.nn.Module):
    """Single GCN layer (supports dense adjacency and features)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        weight_init(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, adj_norm, x):
        support = torch.mm(adj_norm, x)  # (N, in)
        out = torch.mm(support, self.weight)  # (N, out)
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(torch.nn.Module):
    """Two‑layer GCN as described in the paper."""

    def __init__(self, nfeat, nhidden, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhidden)
        self.gc2 = GraphConvolution(nhidden, nclass)
        self.dropout = dropout

    def forward(self, adj_norm, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(adj_norm, x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(adj_norm, x)
        return x  # raw logits


def configure_optimizer(model, lr):
    """Adam optimizer; weight decay applied only to first GCN layer."""
    # Separate parameters for weight decay
    decay_params = list(model.gc1.parameters())
    other_params = list(model.gc2.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": other_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    return optimizer


def mask_loss(logits, labels, mask):
    """Cross‑entropy loss restricted to nodes indicated by mask."""
    loss = F.cross_entropy(logits[mask], labels[mask])
    return loss


def mask_accuracy(logits, labels, mask):
    """Accuracy restricted to nodes indicated by mask."""
    preds = logits[mask].argmax(dim=1)
    correct = (preds == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def evaluate(model, adj_norm, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(adj_norm, features)
        loss = mask_loss(logits, labels, mask)
        acc = mask_accuracy(logits, labels, mask)
    return loss, acc


def train_model(features, adj_norm, labels, train_idx, val_idx, smoke=False):
    num_nodes = features.shape[0]
    nclass = int(labels.max().item() + 1)  # labels are 0..6
    model = GCN(
        nfeat=features.shape[1],
        nhidden=CONFIG["hidden"],
        nclass=nclass,
        dropout=CONFIG["dropout"],
    )
    optimizer = configure_optimizer(model, CONFIG["lr"])

    # masks as boolean tensors
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_idx] = True

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0

    max_epochs = 1 if smoke else CONFIG["epochs"]

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(adj_norm, features)
        loss = mask_loss(logits, labels, train_mask)
        loss.backward()
        optimizer.step()

        # validation
        val_loss, val_acc = evaluate(model, adj_norm, features, labels, val_mask)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if not smoke and epochs_since_improve >= CONFIG["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0 or smoke:
            print(
                f"Epoch {epoch+1:03d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | val acc {val_acc*100:.2f}%"
            )

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # final validation score with best model
    _, final_val_acc = evaluate(model, adj_norm, features, labels, val_mask)
    print(f"Validation accuracy: {final_val_acc*100:.2f}%")
    return model


def main():
    set_seeds()
    data = nio.load()
    features_np = data["features"]  # [2708, 1433] uint8? actually float32
    edges_np = data["edges"]  # [E, 2]
    labels_np = data["labels"]  # [2708] int64, -1 for test
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    # Preprocess features
    features = preprocess_features(features_np)

    # Build adjacency matrix
    adj_norm = build_adjacency(edges_np, features.shape[0])

    # Convert labels to torch (ignore -1, replace with 0 for safety; loss will mask them out)
    labels = torch.from_numpy(labels_np).long()
    # Replace -1 with 0 (they will be masked anyway)
    labels[labels == -1] = 0

    smoke = os.environ.get("SMOKE") == "1"
    model = train_model(
        features,
        adj_norm,
        labels,
        train_idx,
        val_idx,
        smoke=smoke,
    )

    # Produce predictions for all nodes
    model.eval()
    with torch.no_grad():
        logits = model(adj_norm, features)
    preds = logits.argmax(dim=1).cpu().numpy()  # shape [2708]

    # Write predictions (helper will select test_idx and val_idx)
    nio.write_predictions(preds)


if __name__ == "__main__":
    main()
