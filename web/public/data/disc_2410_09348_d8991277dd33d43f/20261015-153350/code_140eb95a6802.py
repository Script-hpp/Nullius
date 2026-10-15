

# Plan step -> implemented in: function or class name
# 1. Load data and set seeds -> main()
# 2. Preprocess adjacency (add self-loop, symmetric normalisation) -> normalize_adj()
# 3. Define 2-layer GCN model -> GCN class
# 4. Train with early stopping on validation -> train_model()
# 5. Compute influence matrix (Personalized PageRank) -> compute_ppr_matrix()
# 6. Self‑training loop with pseudo‑labeling (confidence‑based selection) -> self_training()
# 7. Retrain after each expansion -> train_model()
# 8. Keep best validation iteration and report its test accuracy -> main()
# 9. Output predictions for all nodes -> nio.write_predictions()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# -------------------- Helper: adjacency normalisation --------------------
def normalize_adj(adj_sp):
    """
    adj_sp: torch sparse COO tensor (N, N) with self‑loops already added.
    Returns symmetrically normalised adjacency: D^{-1/2} A D^{-1/2}.
    """
    adj_sp = adj_sp.coalesce()
    indices = adj_sp.indices()
    values = adj_sp.values()
    N = adj_sp.shape[0]

    # compute degrees
    deg = torch.sparse.sum(adj_sp, dim=1).to_dense()  # (N,)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # create D^{-1/2} as sparse diagonal
    inv_sqrt_indices = torch.stack([torch.arange(N, device=indices.device),
                                   torch.arange(N, device=indices.device)])
    inv_sqrt_values = deg_inv_sqrt
    D_inv_sqrt = torch.sparse_coo_tensor(inv_sqrt_indices, inv_sqrt_values, (N, N),
                                         device=indices.device)

    # D^{-1/2} * A * D^{-1/2}
    tmp = torch.sparse.mm(D_inv_sqrt, adj_sp)
    out = torch.sparse.mm(tmp, D_inv_sqrt)
    return out.coalesce()

# -------------------- PPR influence matrix (I - (1-a) * \tilde{A})^{-1} --------------------
def compute_ppr_matrix(adj_norm, alpha=0.85):
    """
    adj_norm: normalised adjacency with self‑loops (torch sparse COO).
    Returns influence matrix = (I - (1-alpha) * adj_norm)^{-1}.
    Implemented via Neumann series (power iteration) for efficiency.
    """
    N = adj_norm.shape[0]
    device = adj_norm.device
    eye = torch.sparse_coo_tensor(
        torch.stack([torch.arange(N, device=device),
                     torch.arange(N, device=device)]),
        torch.ones(N, device=device),
        (N, N), device=device
    )
    # I - (1-alpha) * A
    A_shift = adj_norm.mul(1.0 - alpha)
    M = eye - A_shift  # sparse subtraction
    # Convert to dense for inversion (N=2708 is modest)
    M_dense = M.to_dense()
    # Solve (I - (1-alpha)A) X = I  => X = M^{-1}
    # Use torch.linalg.solve (more stable than explicit inverse)
    X = torch.linalg.solve(M_dense, torch.eye(N, device=device))
    return X  # dense tensor (N, N)

# -------------------- GCN model (2‑layer) --------------------
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = nn.Linear(nfeat, nhid)
        self.gc2 = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        # x: (N, nfeat), adj: (N, N) dense or sparse
        x = self.gc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        # propagate via adjacency
        if adj.is_sparse:
            x = torch.sparse.mm(adj, x)
        else:
            x = adj @ x
        x = self.gc2(x)
        return x

# -------------------- Training function with early stopping --------------------
def train_model(model, features, adj, labels, train_mask, val_mask,
                lr=0.001, weight_decay=5e-4, max_epochs=2000, patience=500):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    best_val = 0.0
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_pred = val_logits[val_mask].argmax(dim=1)
            val_acc = (val_pred == labels[val_mask]).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

# -------------------- Helper functions for IGP sampling (simplified) --------------------
def uniform_sample(indices, K, N):
    """Return N samples, each a subset of indices where each element kept with prob 0.5."""
    result = []
    for _ in range(N):
        mask = torch.rand(len(indices), device=indices.device) < 0.5
        result.append(indices[mask])
    return result

def uniform_sample_less_K(indices, K, N):
    """Return N samples each of size at most K (keep prob 0.5, repeat if too large)."""
    result = []
    for _ in range(N):
        while True:
            mask = torch.rand(len(indices), device=indices.device) < 0.5
            selected = indices[mask]
            if len(selected) <= K:
                result.append(selected)
                break
    return result

def get_sparse_column(sparse_tensor, col_idx):
    """Return column col_idx of a sparse COO tensor as a sparse tensor."""
    assert sparse_tensor.is_coalesced, "Tensor must be coalesced"
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    col_mask = indices[1, :] == col_idx
    col_indices = indices[:, col_mask]
    col_values = values[col_mask]
    # reshape to (N,1)
    row_idx = col_indices[0, :]
    col_idx_tensor = torch.zeros_like(row_idx)
    new_indices = torch.stack([row_idx, col_idx_tensor], dim=0)
    return torch.sparse_coo_tensor(new_indices, col_values,
                                   (sparse_tensor.size(0), 1),
                                   device=sparse_tensor.device)

def scale_sparse_matrix(sparse_mat, scale):
    """Scale a sparse tensor by a scalar."""
    assert sparse_mat.is_coalesced
    return torch.sparse_coo_tensor(sparse_mat._indices(),
                                   sparse_mat._values() * scale,
                                   sparse_mat.size(),
                                   device=sparse_mat.device)

def add_sparse_matrices(mat1, mat2):
    """Add two sparse tensors."""
    assert mat1.is_coalesced and mat2.is_coalesced
    indices = torch.cat([mat1._indices(), mat2._indices()], dim=1)
    values = torch.cat([mat1._values(), mat2._values()])
    mat = torch.sparse_coo_tensor(indices, values, mat1.size())
    return mat.coalesce()

def entropy_utility_efficient(y_probs, idx_sampled, idx_train, idx_train_ag,
                              idx_poss_ag, influence_matrix_candidate,
                              total_prob_origin, influence_submatrix_train):
    """Compute utility as in the original code (simplified)."""
    device = influence_matrix_candidate.device
    if idx_sampled.dim() == 0:
        idx_sampled = idx_sampled.unsqueeze(0)

    sample_prob = torch.sparse.mm(influence_matrix_candidate, y_probs[idx_sampled])
    final_prob = total_prob_origin + sample_prob
    final_prob = torch.softmax(final_prob, dim=1)

    class_prob = torch.mean(final_prob, dim=0)
    class_Ent = -torch.sum(class_prob * torch.log(class_prob + 1e-9))

    individual_entropies = -torch.sum(final_prob * torch.log(final_prob + 1e-9), dim=1)
    individual_Ent = torch.mean(individual_entropies)

    train_prob_prop = torch.sparse.mm(influence_submatrix_train,
                                      y_probs[idx_train_ag])
    train_prob_prop = torch.softmax(train_prob_prop, dim=1)
    y_train_pred = torch.softmax(y_probs[idx_train], dim=1)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    Cross_Ent = criterion(train_prob_prop, y_train_pred)

    return -individual_Ent + class_Ent - Cross_Ent

def get_Banzhaf(output, idx_to_label, sampled_matrix, Utilities, N):
    """Compute Banzhaf scores."""
    Values = torch.full((output.shape[0],), float('-inf'), device=output.device)
    for node_i in idx_to_label:
        contain_mask = torch.tensor([(sampled_matrix[n] == node_i).any()
                                     for n in range(N)], device=output.device)
        if contain_mask.any():
            U_mean_with = torch.mean(Utilities[contain_mask])
        else:
            U_mean_with = torch.tensor(0.0, device=output.device)
        if (~contain_mask).any():
            U_mean_without = torch.mean(Utilities[~contain_mask])
        else:
            U_mean_without = torch.tensor(0.0, device=output.device)
        Values[node_i] = U_mean_with - U_mean_without
    return Values

def IGP_sample_banzhaf(g, output, labels, idx_train, idx_train_ag,
                       idx_unlabeled, influence_matrix, confidence, args,
                       avg_ece_validation):
    """Simplified IGP sampling using Banzhaf (core parts)."""
    t = args.sample_num if hasattr(args, 'sample_num') else 100
    K = args.top if hasattr(args, 'top') else 100
    if args.candidate_num == 0:
        if K < 100:
            SAMPLES_RANGE = 2 * K
        else:
            SAMPLES_RANGE = max(K + 100, K)
    else:
        SAMPLES_RANGE = args.candidate_num

    N = int(SAMPLES_RANGE * t / K)
    conf_score, idx_to_label = torch.topk(confidence, SAMPLES_RANGE)

    if not args.k_union:
        sampled_matrix = uniform_sample(idx_to_label, K, N)
    else:
        sampled_matrix = uniform_sample_less_K(idx_to_label, K, N)

    Utilities = torch.zeros(N, device=output.device)

    # prepare training nodes indices
    idx_train_ag_np = torch.where(idx_train_ag == True)[0]
    idx_train_value = torch.where(idx_train == True)[0]

    # total probability origin from training nodes
    y_probs = output.clone()
    num_class = y_probs.shape[1]
    y_probs[idx_train] = F.one_hot(labels[idx_train], num_class).float()

    # influence submatrix for training nodes
    influence_matrix = influence_matrix.transpose(0, 1)  # as in original
    influence_submatrix = influence_matrix.index_select(1, idx_train_value)
    total_prob_origin = torch.sparse.mm(influence_submatrix, y_probs[idx_train])

    # influence submatrix for train vs pseudo-ag
    influence_matrix_train = influence_matrix.index_select(0, idx_train_value)
    # we will compute inside loop

    for n in range(N):
        idx_sampled = sampled_matrix[n]
        influence_matrix_candidate = influence_matrix.index_select(1, idx_sampled)
        idx_sampled_ag = torch.cat((idx_train_ag_np, idx_sampled), dim=0)
        influence_submatrix_train = influence_matrix_train.index_select(1, idx_sampled_ag)
        Utilities[n] = entropy_utility_efficient(y_probs, idx_sampled, idx_train,
                                                 idx_train_ag_np, idx_sampled_ag,
                                                 influence_matrix_candidate,
                                                 total_prob_origin,
                                                 influence_submatrix_train)

    Values = get_Banzhaf(output, idx_to_label, sampled_matrix, Utilities, N)
    return Values

def get_IGP_idx_game(g, output, labels, idx_train, idx_train_ag,
                     idx_unlabeled, influence_matrix, confidence, args,
                     avg_ece_validation):
    """Select top-k nodes by IGP score."""
    if args.PageRank:
        Values = IGP_sample_banzhaf_conv(g, output, labels, idx_train,
                                         idx_train_ag, idx_unlabeled,
                                         influence_matrix, confidence,
                                         args, avg_ece_validation)
    else:
        Values = IGP_sample_banzhaf(g, output, labels, idx_train,
                                    idx_train_ag, idx_unlabeled,
                                    influence_matrix, confidence,
                                    args, avg_ece_validation)
    Ent, pl_idx = torch.topk(Values, args.top if hasattr(args, 'top') else 100)
    return pl_idx

# -------------------- Self‑training loop (IGP‑BANGS selection) --------------------
def self_training(model_init, features, adj, labels,
                  train_mask, val_mask, test_mask,
                  max_iters=40, top_k=100):
    """
    Returns:
        best_model: model with highest validation accuracy across iterations
        best_val_acc: corresponding validation accuracy
        test_pred: test predictions (logits) from best_model
    """
    N = labels.shape[0]
    device = features.device
    # mutable training mask (starts with original train set)
    train_ag = train_mask.clone()
    # pseudo labels placeholder (only meaningful where train_ag is True)
    pseudo_labels = labels.clone()
    # we will keep track of best validation across iterations
    best_val_acc = -1.0
    best_model_state = None
    best_test_logits = None

    # args-like object for selection
    class Args:
        pass
    args = Args()
    args.sample_num = 100
    args.top = top_k
    args.candidate_num = 0
    args.k_union = False
    args.PageRank = False

    model = model_init
    for it in range(max_iters):
        model.eval()
        with torch.no_grad():
            logits = model(features, adj)
            probs = F.softmax(logits, dim=1)
            conf, pred_label = torch.max(probs, dim=1)

        # mask out already labelled nodes
        conf_masked = conf.clone()
        conf_masked[train_ag] = -1.0  # ensure they are not selected

        # compute influence matrix (PPR) once per iteration (could be cached)
        # we reuse the pre‑computed adj_norm from outer scope; need to pass it.
        # For simplicity, we compute it here using the same function as in main.
        # However to avoid recomputation we will compute outside and pass in.
        # We'll adjust: self_training will receive adj_norm and use it.
        # For now, we'll compute inside (acceptable overhead).
        # Note: we need adj_norm; we will capture from outer scope via closure.
        # To keep the function signature simple, we'll assume adj_norm is global.
        # Instead, we'll modify self_training to accept adj_norm.
        # We'll change the call in main accordingly.
        # For this snippet, we'll assume adj_norm is available as a global variable.
        # In the final script we will pass it as argument.
        # Placeholder:
        influence_matrix = compute_ppr_matrix(adj_norm, alpha=0.85)  # dense

        # select nodes via IGP‑BANGS
        idx_unlabeled = ~train_ag
        pl_idx = get_IGP_idx_game(None, logits, labels, train_mask,
                                  train_ag, idx_unlabeled,
                                  influence_matrix, conf, args, 0.0)
        if pl_idx.numel() == 0:
            break

        # assign pseudo‑labels
        pseudo_labels[pl_idx] = pred_label[pl_idx]
        train_ag[pl_idx] = True

        # retrain model from scratch with updated pseudo‑labels
        model, val_acc = train_model(
            model, features, adj, pseudo_labels,
            train_ag, val_mask,
            lr=0.001, weight_decay=5e-4,
            max_epochs=2000, patience=500
        )

        # evaluate on validation (visible) set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # compute test logits for this model
            with torch.no_grad():
                best_test_logits = model(features, adj)
        # else: keep previous best

    # restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        test_logits = best_test_logits
    else:
        # fallback: use last model
        test_logits = model(features, adj)

    return model, best_val_acc, test_logits

# -------------------- Main --------------------
def main():
    # ----- seeds -----
    seed = int(os.environ.get('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ----- smoke test handling -----
    smoke = os.environ.get('SMOKE', '0') == '1'
    if smoke:
        inner_max_epochs = 1
        inner_patience = 10000  # effectively no early stopping
        outer_max_iters = 2
    else:
        inner_max_epochs = 2000
        inner_patience = 500
        outer_max_iters = 40

    # ----- load data -----
    data = nio.load()  # returns dict with keys: features, edges, labels, train_idx, val_idx, test_idx
    features = torch.from_numpy(data['features']).float()          # (2708,1433)
    edges = torch.from_numpy(data['edges']).long()                 # (E,2)
    labels = torch.from_numpy(data['labels']).long()               # (2708,)
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()

    N = features.shape[0]

    # ----- build adjacency (add self‑loop, symmetric normalisation) -----
    adj = torch.sparse_coo_tensor(
        edges.t(),  # shape (2, E)
        torch.ones(edges.shape[0], dtype=torch.float32, device=features.device),
        (N, N),
        device=features.device
    ).coalesce()
    # add self‑loop
    self_loop = torch.sparse_coo_tensor(
        torch.stack([torch.arange(N, device=features.device),
                     torch.arange(N, device=features.device)]),
        torch.ones(N, dtype=torch.float32, device=features.device),
        (N, N),
        device=features.device
    )
    adj = adj + self_loop
    adj = adj.coalesce()
    # normalise
    adj_norm = normalize_adj(adj)

    # ----- masks -----
    train_mask = torch.zeros(N, dtype=torch.bool, device=features.device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=features.device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=features.device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # ----- independent candidate selection: hide last 20% of validation -----
    # shuffle val_idx with seed, then take first 80% as visible
    val_idx_shuffled = val_idx[torch.randperm(val_idx.size(0), generator=torch.Generator().manual_seed(seed))]
    n_visible = int(0.8 * val_idx_shuffled.size(0))
    visible_val_idx = val_idx_shuffled[:n_visible]
    hidden_val_idx = val_idx_shuffled[n_visible:]  # not used for training or early stopping

    visible_val_mask = torch.zeros(N, dtype=torch.bool, device=features.device)
    visible_val_mask[visible_val_idx] = True

    # ----- initial model -----
    nfeat = features.shape[1]
    nhid = 128          # as per plan
    nclass = int(labels.max().item()) + 1  # 7
    dropout = 0.3
    model = GCN(nfeat, nhid, nclass, dropout).to(features.device)

    # ----- self‑training -----
    # we need to pass adj_norm to self_training; we'll adjust function signature later.
    # For now, we'll use a closure capturing adj_norm.
    # Redefine self_training to accept adj_norm as argument.
    # We'll replace the call below with the updated version.
    # Since we cannot change the function definition after it's used, we'll copy the definition with the extra param.
    # To avoid duplication, we'll just define a new inner function here that uses adj_norm from outer scope.
    def self_training_inner(model_init, features, adj, labels,
                            train_mask, val_mask, test_mask,
                            max_iters, top_k):
        N = labels.shape[0]
        device = features.device
        train_ag = train_mask.clone()
        pseudo_labels = labels.clone()
        best_val_acc = -1.0
        best_model_state = None
        best_test_logits = None

        class Args:
            pass
        args = Args()
        args.sample_num = 100
        args.top = top_k
        args.candidate_num = 0
        args.k_union = False
        args.PageRank = False

        model = model_init
        for it in range(max_iters):
            model.eval()
            with torch.no_grad():
                logits = model(features, adj)
                probs = F.softmax(logits, dim=1)
                conf, pred_label = torch.max(probs, dim=1)

            conf_masked = conf.clone()
            conf_masked[train_ag] = -1.0

            # influence matrix (cached)
            influence_matrix = compute_ppr_matrix(adj_norm, alpha=0.85)

            idx_unlabeled = ~train_ag
            pl_idx = get_IGP_idx_game(None, logits, labels, train_mask,
                                      train_ag, idx_unlabeled,
                                      influence_matrix, conf, args, 0.0)
            if pl_idx.numel() == 0:
                break

            pseudo_labels[pl_idx] = pred_label[pl_idx]
            train_ag[pl_idx] = True

            model, val_acc = train_model(
                model, features, adj, pseudo_labels,
                train_ag, visible_val_mask,  # use only visible validation for early stopping
                lr=0.001, weight_decay=5e-4,
                max_epochs=inner_max_epochs, patience=inner_patience
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                with torch.no_grad():
                    best_test_logits = model(features, adj)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            test_logits = best_test_logits
        else:
            test_logits = model(features, adj)
        return model, best_val_acc, test_logits

    model, best_val_acc, test_logits = self_training_inner(
        model, features, adj, labels,
        train_mask, visible_val_mask, test_mask,
        max_iters=outer_max_iters, top_k=100
    )

    # ----- produce predictions for all nodes -----
    model.eval()
    with torch.no_grad():
        all_logits = model(features, adj_norm)
        # nio.write_predictions accepts either class indices or logits
        # We'll provide logits as per contract
    nio.write_predictions(all_logits.cpu())

    # ----- print final validation accuracy (for sanity) -----
    print(f"Best validation accuracy (visible): {best_val_acc:.4f}")

if __name__ == "__main__":
    main()