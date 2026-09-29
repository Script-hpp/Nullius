import os
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp

def main():
    # ----- seed -----
    seed_str = os.environ.get("SEED", "0")
    try:
        seed = int(seed_str)
    except ValueError:
        seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- load data -----
    data = np.load("/data/cora_public.npz", allow_pickle=True)
    features = data["features"].astype(np.float32)          # [N, F]
    edges = data["edges"]                                   # [E, 2] (already symmetric)
    labels = data["labels"].astype(np.int64)                # [N]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    N = features.shape[0]
    num_classes = int(labels[labels >= 0].max()) + 1

    # ----- build symmetric normalized adjacency with self-loops -----
    # adjacency from edges (already contains both directions)
    adj = sp.coo_matrix((np.ones(edges.shape[0], dtype=np.float32),
                         (edges[:, 0], edges[:, 1])),
                        shape=(N, N), dtype=np.float32)
    # add self-loops
    adj = adj + sp.identity(N, dtype=np.float32)

    # degree matrix
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    # S = D^{-1/2} * A * D^{-1/2}
    S = D_inv_sqrt.dot(adj).dot(D_inv_sqrt)   # sparse matrix

    # ----- feature propagation: S^K X with K=2 -----
    # propagate once
    tmp = S.dot(features)                     # [N, F]
    # propagate twice
    propagated = S.dot(tmp)                   # [N, F]

    Z = torch.from_numpy(propagated)          # torch tensor, float32
    Z_train = Z[train_idx]
    Z_val = Z[val_idx]
    Z_test = Z[test_idx]

    y_train = torch.from_numpy(labels[train_idx]).long()
    y_val = torch.from_numpy(labels[val_idx]).long()

    # ----- hyperparameter: weight decay (simple grid) -----
    weight_decay_candidates = [0.0, 0.001, 0.005, 0.01]
    best_val_acc = -1.0
    best_state = None
    best_wd = None

    # ----- training settings -----
    lr = 0.2
    max_epochs = 100
    smoke = os.environ.get("SMOKE") == "1"
    if smoke:
        max_epochs = 1
        test_subset = test_idx[:20]
        Z_test_sub = Z[test_subset]
    else:
        test_subset = test_idx
        Z_test_sub = Z[test_idx]

    for wd in weight_decay_candidates:
        model = nn.Linear(Z.shape[1], num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        epoch_best_acc = -1.0
        epoch_best_state = None

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(Z_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            # validation
            model.eval()
            with torch.no_grad():
                val_logits = model(Z_val)
                val_pred = val_logits.argmax(dim=1)
                val_acc = (val_pred == y_val).float().mean().item()

            if val_acc > epoch_best_acc:
                epoch_best_acc = val_acc
                epoch_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch_best_acc > best_val_acc:
            best_val_acc = epoch_best_acc
            best_state = epoch_best_state
            best_wd = wd

    # ----- load best model -----
    model = nn.Linear(Z.shape[1], num_classes)
    model.load_state_dict(best_state)
    model.eval()

    # ----- test prediction -----
    with torch.no_grad():
        test_logits = model(Z_test_sub)
        test_pred = test_logits.argmax(dim=1).cpu().numpy()

    # ----- output -----
    os.makedirs("/output", exist_ok=True)
    out_path = "/output/predictions.csv"
    with open(out_path, "w") as f:
        f.write("id,prediction\n")
        for idx, pred in zip(test_subset, test_pred):
            f.write(f"{idx},{pred}\n")

    # ----- print validation accuracy -----
    print(f"Best validation accuracy: {best_val_acc:.5f}")

if __name__ == "__main__":
    main()
