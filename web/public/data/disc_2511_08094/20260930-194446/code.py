import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
    # ----- environment -----
    seed = int(os.getenv('SEED', '0'))
    smoke = os.getenv('SMOKE', '0') == '1'
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- load data -----
    data = np.load('/data/cora_lcc_public.npz')
    features = torch.from_numpy(data['features']).float()          # [N, F]
    labels = torch.from_numpy(data['labels']).long()               # [N], -1 for test
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()

    N, F = features.shape
    num_classes = int(labels[labels >= 0].max().item()) + 1

    # ----- model -----
    hidden_dim = 64
    model = nn.Sequential(
        nn.Linear(F, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # ----- training loop -----
    max_epochs = 500
    patience = 50
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    batch_size = 256

    for epoch in range(1, max_epochs + 1):
        model.train()
        # shuffle training indices each epoch
        perm = train_idx[torch.randperm(len(train_idx))]
        for i in range(0, len(perm), batch_size):
            batch_nodes = perm[i:i+batch_size]
            optimizer.zero_grad()
            out = model(features[batch_nodes])
            loss = criterion(out, labels[batch_nodes])
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_nodes = val_idx[i:i+batch_size]
                out = model(features[batch_nodes])
                pred = out.argmax(dim=1)
                val_correct += (pred == labels[batch_nodes]).sum().item()
                val_total += batch_nodes.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f'Epoch {epoch:03d} | Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- test prediction -----
    model.eval()
    if smoke:
        test_subset = test_idx[:20]
    else:
        test_subset = test_idx
    with torch.no_grad():
        out = model(features[test_subset])
        pred = out.argmax(dim=1).cpu().numpy()

    # ----- write output -----
    os.makedirs('/output', exist_ok=True)
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for node_idx, p in zip(test_subset.tolist(), pred):
            f.write(f'{node_idx},{int(p)}\n')

    print(f'Final validation accuracy: {best_val_acc:.4f}')

if __name__ == '__main__':
    main()
