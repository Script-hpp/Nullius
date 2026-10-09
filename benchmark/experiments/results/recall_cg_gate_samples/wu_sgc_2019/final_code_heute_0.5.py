# Plan step -> implemented in: set_seed
# Plan step -> implemented in: load_data
# Plan step -> implemented in: build_normalized_adjacency
# Plan step -> implemented in: propagate_features
# Plan step -> implemented in: initialize_linear_layer
# Plan step -> implemented in: define_loss
# Plan step -> implemented in: configure_optimizer
# Plan step -> implemented in: training_loop_with_early_stopping
# Plan step -> implemented in: predict_and_write_predictions
# Plan step -> implemented in: print_final_validation_accuracy

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nullius_io as nio


def set_seed():
    """Read SEED from os.environ, set np.random.seed and torch.manual_seed."""
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    """Load Cora data via nullius_io.load() and return tensors."""
    d = nio.load()  # dict with keys: features, edges, labels, train_idx, val_idx, test_idx
    features = torch.from_numpy(d['features']).to(torch.float32)          # [N, F]
    edges = torch.from_numpy(d['edges']).to(torch.long)                  # [E, 2]
    labels = torch.from_numpy(d['labels']).to(torch.long)                # [N]
    train_idx = torch.from_numpy(d['train_idx']).to(torch.long)
    val_idx = torch.from_numpy(d['val_idx']).to(torch.long)
    test_idx = torch.from_numpy(d['test_idx']).to(torch.long)
    return features, edges, labels, train_idx, val_idx, test_idx


def build_normalized_adjacency(edge_index, num_nodes):
    """
    Build Ã = A + I, compute D^{-1/2} Ã D^{-1/2} and return as a sparse torch tensor.
    edge_index: [E, 2] int64 tensor (already contains both directions).
    """
    # add self-loops
    self_loop = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1).repeat(1, 2)
    edge_with_self = torch.cat([edge_index, self_loop], dim=0)          # [E + N, 2]

    # sparse adjacency with ones
    indices = edge_with_self.t()                                        # [2, E']
    values = torch.ones(indices.size(1), dtype=torch.float32)
    A_tilde = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes))
    A_tilde = A_tilde.coalesce()

    # degree vector
    deg = torch.sparse.sum(A_tilde, dim=1).to_dense()                   # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # normalize: Ã_hat = D^{-1/2} Ã D^{-1/2}
    row = indices[0]
    col = indices[1]
    norm_vals = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    A_hat = torch.sparse_coo_tensor(indices, norm_vals, size=(num_nodes, num_nodes))
    A_hat = A_hat.coalesce()
    return A_hat


def propagate_features(A_hat, features, K=2):
    """Compute H = Â^K * features using sparse‑dense multiplication."""
    H = features
    for _ in range(K):
        H = torch.sparse.mm(A_hat, H)
    return H


def initialize_linear_layer(H, num_classes):
    """Initialize a single linear layer W: Linear(H.shape[1], num_classes, bias=False)."""
    return nn.Linear(H.shape[1], num_classes, bias=False)


def define_loss():
    """Return CrossEntropyLoss."""
    return nn.CrossEntropyLoss()


def configure_optimizer(model, lr=0.2, weight_decay=5e-4):
    """Return Adam optimizer."""
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def training_loop_with_early_stopping(model, H, labels, train_idx, val_idx,
                                      optimizer, loss_fn,
                                      max_epochs=200, patience=50):
    """Train model with validation‑based early stopping; return best val accuracy."""
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(H)
        loss = loss_fn(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(H)
            val_pred = val_logits[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc


def predict_and_write_predictions(model, H):
    """Run inference on all nodes, get predicted class, and write via nullius_io."""
    model.eval()
    with torch.no_grad():
        logits = model(H)
        preds = logits.argmax(dim=1)               # [N] int64
    # Convert to numpy array as required by helper
    nio.write_predictions(preds.cpu().numpy())
    return preds


def print_final_validation_accuracy(val_acc):
    """Print the final validation accuracy to stdout."""
    print(f'Validation accuracy: {val_acc:.4f}')


def main():
    set_seed()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()

    num_nodes = features.shape[0]
    num_classes = int(labels.max().item()) + 1  # labels are 0..6, ignoring -1 entries

    # Build normalized adjacency Â
    A_hat = build_normalized_adjacency(edges, num_nodes)

    # Propagate features (K=2 as per plan)
    H = propagate_features(A_hat, features, K=2)

    # Model, loss, optimizer
    model = initialize_linear_layer(H, num_classes)
    loss_fn = define_loss()
    lr = 0.2
    weight_decay = 5e-4
    optimizer = configure_optimizer(model, lr, weight_decay)

    # Determine training duration based on SMOKE mode
    smoke = os.environ.get('SMOKE') == '1'
    max_epochs = 1 if smoke else 200
    patience = 50  # unchanged; with 1 epoch it will not trigger early stop

    # Train
    best_val_acc = training_loop_with_early_stopping(
        model, H, labels, train_idx, val_idx,
        optimizer, loss_fn,
        max_epochs=max_epochs, patience=patience
    )

    # Predict and write outputs
    predict_and_write_predictions(model, H)

    # Print final validation accuracy
    print_final_validation_accuracy(best_val_acc)


if __name__ == '__main__':
    main()
