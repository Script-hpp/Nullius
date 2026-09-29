import os
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp

def main():
    # ----- Seed -----
    seed = int(os.getenv('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- Smoke mode -----
    smoke = os.getenv('SMOKE', '0') == '1'

    # ----- Load data -----
    data = np.load('/data/cora_public.npz')
    features = data['features'].astype(np.float32)          # [N, F]
    edges = data['edges']                                   # [E, 2] (both directions)
    labels = data['labels'].astype(np.int64)                # [N], -1 for test
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    N = features.shape[0]

    # ----- Build adjacency matrix (sparse) -----
    adj = sp.csr_matrix(
        (np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(N, N)
    )

    # ----- Propagation function (S^K X) -----
    def propagate(X, K=2):
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0   # avoid division by zero
        X_temp = X * d_inv_sqrt[:, None]         # right multiply D^{-1/2}
        for _ in range(K):
            X_temp = adj @ X_temp                # A * (D^{-1/2} X)
            X_temp = X_temp * d_inv_sqrt[:, None]# left multiply D^{-1/2}
        return X_temp

    # ----- Get propagated features -----
    Z_np = propagate(features, K=2)               # [N, F]
    Z = torch.from_numpy(Z_np)                    # torch float32
    labels_t = torch.from_numpy(labels)           # torch int64

    num_classes = int(labels.max().item()) + 1    # assumes labels 0..6

    # ----- Simple logistic regression model -----
    class LogReg(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
        def forward(self, x):
            return self.linear(x)

    model = LogReg(Z.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss()

    # ----- Training helper -----
    def train_model(weight_decay, max_epochs, patience):
        # reinitialize parameters
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=weight_decay)

        best_val_acc = 0.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(Z[train_idx])
            loss = criterion(out, labels_t[train_idx])
            loss.backward()
            optimizer.step()

            # validation
            model.eval()
            with torch.no_grad():
                val_out = model(Z[val_idx])
                val_pred = val_out.argmax(dim=1)
                val_acc = (val_pred == labels_t[val_idx]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return best_val_acc

    # ----- Smoke mode: 1 epoch, first 20 test nodes -----
    if smoke:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=0.0)
        model.train()
        optimizer.zero_grad()
        out = model(Z[train_idx])
        loss = criterion(out, labels_t[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(Z[val_idx])
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred == labels_t[val_idx]).float().mean().item()
        print(f"Validation accuracy: {val_acc:.4f}")

        test_sub = test_idx[:20]
        with torch.no_grad():
            test_out = model(Z[test_sub])
            preds = test_out.argmax(dim=1).cpu().numpy()

        os.makedirs('/output', exist_ok=True)
        with open('/output/predictions.csv', 'w') as f:
            f.write('id,prediction\n')
            for idx, pred in zip(test_sub, preds):
                f.write(f'{int(idx)},{int(pred)}\n')
        return

    # ----- Hyperopt search for weight_decay (60 trials) -----
    best_wd = None
    best_val = -1.0
    low, high = 1e-6, 1e-1
    log_low, log_high = np.log(low), np.log(high)

    for trial in range(60):
        log_wd = np.random.uniform(log_low, log_high)
        wd = np.exp(log_wd)
        val_acc = train_model(wd, max_epochs=30, patience=5)
        if val_acc > best_val:
            best_val = val_acc
            best_wd = wd

    # ----- Final training with selected weight_decay -----
    final_val_acc = train_model(best_wd, max_epochs=100, patience=10)
    print(f"Validation accuracy: {final_val_acc:.4f}")

    # ----- Predict on full test set -----
    model.eval()
    with torch.no_grad():
        test_out = model(Z[test_idx])
        preds = test_out.argmax(dim=1).cpu().numpy()

    os.makedirs('/output', exist_ok=True)
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx, pred in zip(test_idx, preds):
            f.write(f'{int(idx)},{int(pred)}\n')

if __name__ == '__main__':
    main()
