# Plan step -> implemented in: set_seeds
# Plan step -> implemented in: load_data
# Plan step -> implemented in: GATLayer
# Plan step -> implemented in: GATNet
# Plan step -> implemented in: setup_optimizer
# Plan step -> implemented in: train_model
# Plan step -> implemented in: predict_and_save

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

def set_seeds():
    seed = int(os.getenv('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    data = np.load('/data/cora_public.npz')
    features = torch.from_numpy(data['features']).float()          # [N, F]
    edges = torch.from_numpy(data['edges']).long()                # [E, 2] (u, v)
    labels = torch.from_numpy(data['labels']).long()              # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.6, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, num_heads * out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.attn = nn.Parameter(torch.empty(size=(num_heads, 2 * out_features)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edge_index):
        # x: [N, in_features], edge_index: [2, E]
        if self.training:
            x = F.dropout(x, self.dropout, training=True)

        N = x.size(0)
        Wh = torch.matmul(x, self.W)          # [N, num_heads * out_features]
        Wh = Wh.view(N, self.num_heads, self.out_features)  # [N, heads, out]

        edge_src, edge_dst = edge_index[0], edge_index[1]  # [E], [E]
        Wh_src = Wh[edge_src]                     # [E, heads, out]
        Wh_dst = Wh[edge_dst]                     # [E, heads, out]

        # concat -> [E, heads, 2*out]
        concat = torch.cat([Wh_src, Wh_dst], dim=-1)
        # attention scores -> [E, heads]
        e = self.leakyrelu((concat * self.attn).sum(dim=-1))

        # softmax per source node
        alpha = torch.zeros_like(e)
        for head in range(self.num_heads):
            e_head = e[:, head]
            src_nodes = edge_src.unique()
            for src in src_nodes:
                mask = (edge_src == src)
                if not mask.any():
                    continue
                e_src = e_head[mask]
                e_max = e_src.max()
                exp_e = torch.exp(e_src - e_max)
                sum_exp = exp_e.sum()
                alpha_src = exp_e / sum_exp
                alpha[mask, head] = alpha_src

        # dropout on attention coefficients
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # weighted sum of neighbor features
        out = torch.zeros(N, self.num_heads, self.out_features, device=x.device)
        for head in range(self.num_heads):
            alpha_head = alpha[:, head]                     # [E]
            weighted = Wh_dst[:, head, :] * alpha_head.unsqueeze(-1)  # [E, out]
            out.index_add_(0, edge_dst, weighted)           # accumulate to destination nodes

        out = out.view(N, self.num_heads * self.out_features)  # [N, heads*out]
        if self.concat:
            return out
        else:
            return out.mean(dim=1)

class GATNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, heads=8):
        super(GATNet, self).__init__()
        self.dropout = dropout
        self.gat1 = GATLayer(nfeat, nhid // heads, heads, dropout=dropout, concat=True)
        self.gat2 = GATLayer(nhid, nclass, 1, dropout=dropout, concat=True)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

def setup_optimizer(model, lr=0.005, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def train_model(model, features, edge_index, labels, train_idx, val_idx, optimizer, patience=100, max_epochs=10000, smoke=False):
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features, edge_index)
            val_loss = F.cross_entropy(val_logits[val_idx], labels[val_idx])
            val_pred = val_logits[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if smoke and epoch >= 0:  # smoke mode runs exactly 1 epoch
            break
        if epochs_no_improve >= patience:
            break

    # restore best model
    model.load_state_dict(best_state)
    # final validation accuracy with best model
    model.eval()
    with torch.no_grad():
        val_logits = model(features, edge_index)
        val_pred = val_logits[val_idx].argmax(dim=1)
        val_acc = (val_pred == labels[val_idx]).float().mean().item()
    print(f'Validation Accuracy: {val_acc:.4f}')
    return model

def predict_and_save(model, features, edge_index, test_idx, val_idx, smoke=False):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        preds = logits.argmax(dim=1)

    # test predictions
    if smoke:
        test_ids = test_idx[:20]
    else:
        test_ids = test_idx
    with open('/output/predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'prediction'])
        for node_id, pred in zip(test_ids.tolist(), preds[test_ids].tolist()):
            writer.writerow([node_id, pred])

    # validation predictions (for selection)
    if smoke:
        val_ids = val_idx[:20]
    else:
        val_ids = val_idx
    with open('/output/val_predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'prediction'])
        for node_id, pred in zip(val_ids.tolist(), preds[val_ids].tolist()):
            writer.writerow([node_id, pred])

def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    # edge_index shape [2, E]
    edge_index = edges.t().contiguous()  # [2, E]

    nfeat = features.size(1)
    nclass = int(labels.max().item() + 1)  # labels are 0..6, -1 for test but max ignores -1? safer:
    # compute number of classes from known labels (excluding -1)
    known_labels = labels[labels >= 0]
    nclass = int(known_labels.max().item() + 1)

    model = GATNet(nfeat=nfeat, nhid=64, nclass=nclass, dropout=0.6, heads=8)
    optimizer = setup_optimizer(model, lr=0.005, weight_decay=5e-4)

    smoke = os.getenv('SMOKE') == '1'
    model = train_model(model, features, edge_index, labels, train_idx, val_idx,
                        optimizer, patience=100, max_epochs=10000, smoke=smoke)
    predict_and_save(model, features, edge_index, test_idx, val_idx, smoke=smoke)

if __name__ == '__main__':
    main()
