# Plan step -> imported libraries
import os
import numpy as np
import torch
import torch.nn.functional as F
import nullius_io as nio

# Plan step -> hyperparameter container
CONFIG = {"lr": 0.01, "epochs": 200, "dropout": 0.5, "hidden": 16, "patience": 10}
CONFIG["weight_decay"] = 5e-4  # L2 penalty on first-layer weights

def main():
    # Plan step -> seeding
    seed = int(os.getenv("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Plan step -> load data
    data = nio.load()
    features = data['features']          # np.float32 [N, F]
    edges = data['edges']                # np.int64 [E, 2] (both directions)
    labels = data['labels']              # np.int64 [N] (-1 for test)
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    N = features.shape[0]

    # Plan step -> row-normalize features
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    features_norm = features / row_sum
    features_t = torch.from_numpy(features_norm).float()

    # Plan step -> build adjacency matrix with self-loops and symmetric normalization
    edge_index = torch.from_numpy(edges.T).long()               # [2, E]
    self_loop = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    edge_index_with_self = torch.cat([edge_index, self_loop], dim=1)  # [2, E+N]
    values = torch.ones(edge_index_with_self.shape[1], dtype=torch.float32)

    deg = torch.zeros(N, dtype=torch.float32)
    deg = deg.scatter_add(0, edge_index_with_self[0], values)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    norm_values = deg_inv_sqrt[edge_index_with_self[0]] * deg_inv_sqrt[edge_index_with_self[1]] * values

    adj = torch.sparse_coo_tensor(
        edge_index_with_self,
        norm_values,
        size=(N, N),
        dtype=torch.float32
    ).coalesce()

    # Plan step -> model parameters
    input_dim = features_t.shape[1]
    hidden_dim = CONFIG["hidden"]
    output_dim = int(labels[labels != -1].max() + 1)  # assumes 0..5 -> 6

    # Glorot initialization
    W1 = torch.nn.Parameter(torch.empty(input_dim, hidden_dim))
    W2 = torch.nn.Parameter(torch.empty(hidden_dim, output_dim))
    torch.nn.init.xavier_uniform_(W1)
    torch.nn.init.xavier_uniform_(W2)

    optimizer = torch.optim.Adam([W1, W2], lr=CONFIG["lr"])
    weight_decay = CONFIG["weight_decay"]

    # Masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(N, dtype=torch.bool)
    val_mask[val_idx] = True

    labels_long = torch.from_numpy(labels).long()
    labels_long[labels_long == -1] = 0  # dummy value, will be masked

    def forward(x, adj_mat, training=False):
        x = F.dropout(x, p=CONFIG["dropout"], training=training)
        support = torch.sparse.mm(adj_mat, x)
        x = torch.mm(support, W1)
        x = F.relu(x)
        x = F.dropout(x, p=CONFIG["dropout"], training=training)
        support = torch.sparse.mm(adj_mat, x)
        x = torch.mm(support, W2)
        return x

    # Plan step -> training loop with early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = 1 if os.getenv("SMOKE") == "1" else CONFIG["epochs"]
    val_acc_last = 0.0

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = forward(features_t, adj, training=True)
        loss_train = F.cross_entropy(logits[train_mask], labels_long[train_mask])
        loss = loss_train + weight_decay * torch.sum(W1 ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            logits_eval = forward(features_t, adj, training=False)
            loss_val = F.cross_entropy(logits_eval[val_mask], labels_long[val_mask])
            preds_val = logits_eval[val_mask].argmax(dim=1)
            acc_val = (preds_val == labels_long[val_mask]).float().mean().item()

        print(
            f"Epoch {epoch+1}: train_loss={loss_train.item():.5f}, "
            f"val_loss={loss_val.item():.5f}, val_acc={acc_val:.5f}"
        )

        if epoch >= CONFIG["patience"]:
            if loss_val.item() > best_val_loss:
                epochs_no_improve += 1
            else:
                best_val_loss = loss_val.item()
                epochs_no_improve = 0
            if epochs_no_improve >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
        val_acc_last = acc_val

    # Plan step -> final predictions for all nodes
    with torch.no_grad():
        logits_all = forward(features_t, adj, training=False)
    nio.write_predictions(logits_all.cpu().numpy())
    print(f"Validation accuracy: {val_acc_last:.5f}")

if __name__ == "__main__":
    main()
