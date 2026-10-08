

# Plan step -> implemented in: set_seed function
# Plan step -> implemented in: load_data function
# Plan step -> implemented in: GATLayer class
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
    Uses dense attention matrix for simplicity (N is small).
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
        self.W = nn.Parameter(torch.empty(size=(num_heads, in_features, out_features)))
        # Attention weight vector for each head: shape (num_heads, 2 * out_features)
        self.a = nn.Parameter(torch.empty(size=(num_heads, 2 * out_features)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, edge_index):
        """
        h: (N, in_features)
        edge_index: LongTensor of shape [2, E] (src, dst)
        Returns: (N, out_features') where out_features' = num_heads * out_features if concat else out_features
        """
        N = h.size(0)
        # Apply dropout to input features
        h = F.dropout(h, self.dropout, training=self.training)

        # Linear transformation per head: list of (N, out_features)
        Wh_list = [torch.matmul(h, self.W[head]) for head in range(self.num_heads)]  # each (N, Fout)
        Wh = torch.stack(Wh_list, dim=0)  # (H, N, Fout)

        # Prepare source and destination indices
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        E = src.size(0)

        # Compute attention scores per head
        # Wh_src: (H, E, Fout), Wh_dst: (H, E, Fout)
        Wh_src = Wh[:, src, :]  # (H, E, Fout)
        Wh_dst = Wh[:, dst, :]  # (H, E, Fout)
        # Concatenate [Wh_i || Wh_j] per head -> (H, E, 2*Fout)
        concat_feat = torch.cat([Wh_src, Wh_dst], dim=-1)  # (H, E, 2*Fout)

        # Compute e = LeakyReLU(a^T * concat_feat)
        # self.a: (H, 2*Fout) -> unsqueeze to (H,1,2*F) for broadcasting
        a_expanded = self.a.unsqueeze(1)  # (H, 1, 2*Fout)
        e = torch.sum(a_expanded * concat_feat, dim=-1)  # (H, E)
        e = self.leakyrelu(e)

        # Convert to dense attention matrix per head and mask non-edges
        alpha_list = []
        for head in range(self.num_heads):
            attn = torch.zeros(N, N, device=h.device)  # (N, N)
            attn[src, dst] = e[head]  # fill edges
            # Mask: keep only edges, set others to -inf for softmax
            mask = torch.zeros(N, N, device=h.device, dtype=torch.bool)
            mask[src, dst] = True
            neg_inf = torch.tensor(-9e15, device=h.device)
            attn = torch.where(mask, attn, neg_inf)
            # Softmax over destination dimension (dim=1) -> each row sums to 1
            attn = F.softmax(attn, dim=1)
            # Apply dropout to attention coefficients
            attn = F.dropout(attn, self.dropout, training=self.training)
            alpha_list.append(attn)
        alpha = torch.stack(alpha_list, dim=0)  # (H, N, N)

        # Compute node features per head: (H, N, Fout) = alpha @ Wh
        # Wh: (H, N, Fout) -> we need to transpose for batch matmul? Actually alpha (H,N,N) @ Wh (H,N,Fout) -> (H,N,Fout)
        h_prime = torch.matmul(alpha, Wh)  # (H, N, Fout)

        if self.concat:
            # Concatenate heads -> (N, H * Fout)
            h_prime = h_prime.transpose(0, 1).contiguous().view(N, -1)
        else:
            # Average heads -> (N, Fout)
            h_prime = h_prime.mean(dim=0)

        return h_prime

class GATModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """
        nhid: total hidden dimension after first layer (concat=True).
        Split into nheads heads each of size nhid // nheads.
        """
        super(GATModel, self).__init__()
        self.dropout = dropout
        assert nhid % nheads == 0
        nhid_per_head = nhid // nheads
        self.attentions = nn.ModuleList()
        for _ in range(nheads):
            self.attentions.append(
                GATLayer(nfeat, nhid_per_head, 1, dropout, alpha, concat=False)  # each head outputs nhid_per_head
            )
        # After first layer we concat -> nhid
        self.out_att = GATLayer(nhid, nclass, 1, dropout, alpha, concat=False)  # single head for output

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply each attention head and concat
        head_outs = [att(x, edge_index) for att in self.attentions]
        x = torch.cat(head_outs, dim=1)  # (N, nhid)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))  # (N, nclass)
        return x

def accuracy(output, labels):
    preds = output.argmax(dim=1)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)

def train(model, features, edge_index, labels, train_idx, val_idx, optimizer, epochs, patience, smoke=False):
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index)
        loss_train = F.cross_entropy(output[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, edge_index)
            loss_val = F.cross_entropy(output[val_idx], labels[val_idx])
            acc_val = accuracy(output[val_idx], labels[val_idx])

        if smoke:
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
        output = model(features, edge_index)
        acc_val = accuracy(output[val_idx], labels[val_idx])
    print(f'Final validation accuracy: {acc_val.item():.4f}')
    return acc_val.item()

def predict(model, features, edge_index, idx):
    model.eval()
    with torch.no_grad():
        output = model(features, edge_index)
        pred = output[idx].argmax(dim=1)
    return pred.cpu().numpy()

def save_predictions(csv_path, indices, predictions):
    with open(csv_path, 'w') as f:
        f.write('id,prediction\n')
        for idx, pred in zip(indices, predictions):
            f.write(f'{int(idx)},{int(pred)}\n')

def main():
    seed = int(os.getenv('SEED', '0'))
    set_seed(seed)

    smoke = os.getenv('SMOKE', '0') == '1'

    features, edges, labels, train_idx, val_idx, test_idx = load_data('/data/cora_public.npz')

    # Build edge_index (src, dst) from edges tensor [E,2]
    edge_index = edges.t()  # shape [2, E]

    N = features.size(0)
    # Determine number of classes from train+val labels (ignore -1)
    known_labels = torch.cat([labels[train_idx], labels[val_idx]]).unique()
    nclass = known_labels.size(0)

    # Hyperparameters as per plan (adjusted for speed if needed)
    nfeat = features.size(1)
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

    val_acc = train(model, features, edge_index, labels, train_idx, val_idx, optimizer, epochs, patience, smoke=smoke)

    # Predict on test (or subset in smoke)
    if smoke:
        test_subset = test_idx[:20]
        pred_test = predict(model, features, edge_index, test_subset)
        save_predictions('/output/predictions.csv', test_subset, pred_test)
        # Validation predictions: first 20 of sorted val_idx
        val_sorted, _ = torch.sort(val_idx)
        val_subset = val_sorted[:20]
        pred_val = predict(model, features, edge_index, val_subset)
        save_predictions('/output/val_predictions.csv', val_subset, pred_val)
    else:
        pred_test = predict(model, features, edge_index, test_idx)
        save_predictions('/output/predictions.csv', test_idx, pred_test)
        pred_val = predict(model, features, edge_index, val_idx)
        save_predictions('/output/val_predictions.csv', val_idx, pred_val)

if __name__ == '__main__':
    main()