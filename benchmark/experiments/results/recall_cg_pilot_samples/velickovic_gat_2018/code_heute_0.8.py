# Plan step -> implemented in: set_seed function
# Plan step -> implemented in: load_data function
# Plan step -> implemented in: GATModel class
# Plan step -> implemented in: train function (loss, optimizer, training loop)
# Plan step -> implemented in: predict function
# Plan step -> implemented in: save_predictions function
# Plan step -> implemented in: train function (prints final validation accuracy)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    features = torch.FloatTensor(data['features'])
    edges = torch.LongTensor(data['edges'])  # shape [E, 2]
    labels = torch.LongTensor(data['labels'])
    train_idx = torch.LongTensor(data['train_idx'])
    val_idx = torch.LongTensor(data['val_idx'])
    test_idx = torch.LongTensor(data['test_idx'])
    return features, edges, labels, train_idx, val_idx, test_idx

class GATLayer(nn.Module):
    """
    Single GAT layer (multi-head) with optional concat.
    """
    def __init__(self, in_features, out_features, num_heads, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha

        # Weight matrix for each head: shape (num_heads, in_features, out_features)
        self.W = Parameter(torch.empty(size=(num_heads, in_features, out_features)))
        # Attention weight vector for each head: shape (num_heads, 2 * out_features)
        self.a = Parameter(torch.empty(size=(num_heads, 2 * out_features)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier init for W and a
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        """
        h: (N, in_features)
        adj: sparse COO tensor (N, N) with edge indices
        Returns: (N, out_features') where out_features' = num_heads * out_features if concat else out_features
        """
        N = h.size(0)
        # Linear transformation per head: (num_heads, N, out_features)
        Wh = torch.einsum('nij,bj->bni', self.W, h)  # (num_heads, N, out_features)

        # Prepare for attention: need Wh_i and Wh_j for each edge
        edge_src = adj.indices()[0]  # (E,)
        edge_dst = adj.indices()[1]  # (E,)

        # Wh per head for src and dst
        Wh_src = Wh[:, edge_src, :]  # (num_heads, E, out_features)
        Wh_dst = Wh[:, edge_dst, :]  # (num_heads, E, out_features)

        # Concatenate [Wh_i || Wh_j] per head
        concat_feat = torch.cat([Wh_src, Wh_dst], dim=-1)  # (num_heads, E, 2*out_features)

        # Compute unnormalized attention scores: e = LeakyReLU(a^T * concat_feat)
        e = self.leakyrelu(torch.einsum('bei,bei->be', self.a, concat_feat))  # (num_heads, E)

        # Convert to dense attention matrix per head for softmax normalization
        # We'll compute softmax over neighbors of each src node.
        # First, convert e to same shape as adj values per head.
        # Since we have multiple heads, we will compute attention per head separately.
        # We'll create a sparse tensor per head with values = e[head]
        alpha = []
        for head in range(self.num_heads):
            # Sparse tensor of shape (N, N) with values e[head]
            attn_head = torch.sparse_coo_tensor(adj.indices(), e[head], size=(N, N))
            # Softmax over columns (neighbors of src) -> we need to normalize per src node.
            # torch.sparse.softmax does not exist; we implement manually:
            # For each src node, compute exp(e) and sum over its neighbors.
            # We'll use scatter_add via torch.sparse.
            # Convert to dense for simplicity (N is small 2708) -> acceptable.
            attn_dense = attn_head.to_dense()
            # Mask non-edges with -inf before softmax
            zero_vec = -9e15 * torch.ones_like(attn_dense)
            attn_dense = torch.where(attn_dense > 0, attn_dense, zero_vec)
            attn_dense = F.softmax(attn_dense, dim=1)  # normalize per src node (row)
            alpha.append(attn_dense)

        alpha = torch.stack(alpha, dim=0)  # (num_heads, N, N)

        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # Compute node features per head: (num_heads, N, out_features)
        h_prime = torch.einsum('hin, hij -> hnj', Wh, alpha)  # (num_heads, N, out_features)

        if self.concat:
            # Concatenate heads -> (N, num_heads * out_features)
            h_prime = h_prime.transpose(0, 1).contiguous().view(N, -1)
        else:
            # Average heads -> (N, out_features)
            h_prime = h_prime.mean(dim=0)

        return h_prime

class GATModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """
        nhid: hidden dimension per head * num_heads? We'll set nhid as total hidden dim.
        We'll split nhid into heads.
        """
        super(GATModel, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        # First layer: split nhid into nheads heads, each head output = nhid // nheads
        assert nhid % nheads == 0
        nhid_per_head = nhid // nheads
        for _ in range(nheads):
            self.attentions.append(
                GATLayer(nfeat, nhid_per_head, 1, dropout, alpha, concat=False)  # each head outputs nhid_per_head
            )
        # After first layer, we concat -> nhid
        self.out_att = GATLayer(nhid, nclass, 1, dropout, alpha, concat=False)  # single head for output

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply each attention head and concat
        head_outs = [att(x, adj) for att in self.attentions]
        x = torch.cat(head_outs, dim=1)  # (N, nhid)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))  # (N, nclass)
        return x

def accuracy(output, labels):
    preds = output.argmax(dim=1)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)

def train(model, features, adj, labels, train_idx, val_idx, optimizer, epochs, patience, smoke=False):
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.cross_entropy(output[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.cross_entropy(output[val_idx], labels[val_idx])
            acc_val = accuracy(output[val_idx], labels[val_idx])

        if smoke:
            # In smoke mode we just run one epoch and break after logging
            print(f'Smoke Epoch: {epoch+1:03d}, loss_train: {loss_train.item():.4f}, loss_val: {loss_val.item():.4f}, acc_val: {acc_val.item():.4f}')
            break

        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 100 == 0:
            print(f'Epoch: {epoch+1:04d}, loss_train: {loss_train.item():.4f}, loss_val: {loss_val.item():.4f}, acc_val: {acc_val.item():.4f}')

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation accuracy with best model
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        acc_val = accuracy(output[val_idx], labels[val_idx])
    print(f'Final validation accuracy: {acc_val.item():.4f}')
    return acc_val.item()

def predict(model, features, adj, idx):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        pred = output[idx].argmax(dim=1)
    return pred.cpu().numpy()

def save_predictions(csv_path, indices, predictions):
    # indices: tensor or list of node ids
    # predictions: numpy array same length
    with open(csv_path, 'w') as f:
        f.write('id,prediction\n')
        for idx, pred in zip(indices, predictions):
            f.write(f'{int(idx)},{int(pred)}\n')

def main():
    seed = int(os.getenv('SEED', '0'))
    set_seed(seed)

    smoke = os.getenv('SMOKE', '0') == '1'

    features, edges, labels, train_idx, val_idx, test_idx = load_data('/data/cora_public.npz')

    # Build adjacency matrix (sparse) from edges
    N = features.size(0)
    adj = torch.sparse_coo_tensor(edges.t(), torch.ones(edges.size(0)), size=(N, N))
    # Ensure coalesced (sum duplicates)
    adj = adj.coalesce()

    # Model hyperparameters (as per plan, with some adjustments for speed)
    nfeat = features.size(1)
    nclass = int(labels.max().item() + 1)  # labels are 0..6, -1 for test but max ignores -1? We'll compute from train+val
    # Safer: compute number of unique labels in train and val
    known_labels = torch.cat([labels[train_idx], labels[val_idx]]).unique()
    nclass = known_labels.size(0)

    nhid = 64          # total hidden dimension
    nheads = 8         # number of attention heads in first layer
    dropout = 0.6
    alpha = 0.2        # leakyrelu alpha
    lr = 0.005
    weight_decay = 5e-4
    epochs = 10000
    patience = 100

    model = GATModel(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=dropout, alpha=alpha, nheads=nheads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    val_acc = train(model, features, adj, labels, train_idx, val_idx, optimizer, epochs, patience, smoke=smoke)

    # Predict on test (or subset in smoke)
    if smoke:
        # Only first 20 nodes of test_idx
        test_subset = test_idx[:20]
        pred_test = predict(model, features, adj, test_subset)
        save_predictions('/output/predictions.csv', test_subset, pred_test)
        # Also val predictions: first 20 of sorted val_idx
        val_sorted, _ = torch.sort(val_idx)
        val_subset = val_sorted[:20]
        pred_val = predict(model, features, adj, val_subset)
        save_predictions('/output/val_predictions.csv', val_subset, pred_val)
    else:
        pred_test = predict(model, features, adj, test_idx)
        save_predictions('/output/predictions.csv', test_idx, pred_test)
        pred_val = predict(model, features, adj, val_idx)
        save_predictions('/output/val_predictions.csv', val_idx, pred_val)

if __name__ == '__main__':
    main()
