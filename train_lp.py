from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os
import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, mask_test_edges, get_roc_score, sparse_mx_to_torch_sparse_tensor, normalize_adj
from model import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    help='Dataset to use (cora, citeseer).')
parser.add_argument('--layers', type=int, default=2,
                    help='Number of GCN layers.')
parser.add_argument('--save_dir', type=str, default="plots_lp",
                    help='Directory to save plots/logs.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# Note: Link prediction usually works on the graph structure. 
# We load the data, then split edges.
# We don't use load_data's default normalization because we need to split edges first.
# So we need to access the raw adjacency from utils? 
# load_data returns processed tensors. 
# We might need to refactor or just load raw here. 
# Re-using load_data but we need the scipy sparse matrix before tensor conversion.
# Let's modify load_data slightly or just copy the loading part or accept that we load it then reverse it?
# The `mask_test_edges` takes scipy sparse matrix.
# Let's peek at load_data in utils.py. It returns adj as sparse tensor.
# We should probably modify load_data to return raw if requested or just duplicate simple loading here for LP.
# For simplicity and avoiding breaking changes, let's just do a quick load here similar to utils.

def load_data_raw(path="./", dataset="cora"):
    print('Loading {} dataset for Link Prediction...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}/{}/{}.content".format(path, dataset, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}/{}/{}.cites".format(path, dataset, dataset),
                                    dtype=str)
    edges = np.array(list(map(lambda x: idx_map[x] if x in idx_map else -1, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    edges = edges[(edges[:,0] != -1) & (edges[:,1] != -1)]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)
    # Symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj, features

adj, features = load_data_raw(dataset=args.dataset)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

# Normalize features
from utils import normalize
features = normalize(features)
features = torch.FloatTensor(np.array(features.todense()))

# Normalize training graph
adj_norm = normalize_adj(adj_train + sp.eye(adj_train.shape[0]))
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = torch.FloatTensor(adj_label.toarray())

adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

# Loss function: Binary Cross Entropy with Logits
# We need to reconstruct the adjacency matrix. 
# Z = GCN(A, X)
# A_pred = sigmoid(Z * Z^T)
# Loss = BCE(A_pred, A_true) + KL (if VGAE, but here plain GCN AE)
# Standard GAE loss: 
pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)

# Model
# Output dimension matches hidden for dot product? Or separate? 
# Usually Z is low dim.
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=args.hidden, # Output embedding dimension
            dropout=args.dropout,
            n_layers=args.layers,
            use_pairnorm=False) # PairNorm might hurt reconstruction? Let's keep it simple.

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj_norm = adj_norm.cuda()
    adj_label = adj_label.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(1)

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].item()))
        neg.append(0)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([pos, neg])
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

loss_history = []
val_roc_history = []

for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    z = model(features, adj_norm, return_embeds=True)
    # z is log_softmax in existing GCN model... 
    # WAIT! The GCN model in model.py returns F.log_softmax(x, dim=1).
    # For Link Prediction, we need linear embeddings, NOT softmax probabilities!
    # I need to modify model.py to allow returning embeddings or linear output.
    # OR simply use the GCN_PPI model which returns X directly (but has sigmoid? No, GCN_PPI returns raw logits in my impl? 
    # Let's check model.py)
    
    # model.py:
    # GCN: returns F.log_softmax(x, dim=1)
    # GCN_PPI: returns x
    
    # I should use GCN_PPI class or modify GCN to have an option.
    # Let's use GCN_PPI class structure or create a GCN_Emb class in model.py?
    # Or just modify GCN to take mode='classification' or 'embedding'.
    
    # For now, let's assume I will fix model.py in the next step.
    # Continuing on the assumption `model` returns embeddings `z`.
    
    reconstruction = torch.mm(z, z.t()) # Dot product
    # Reshape for BCE
    # But full matrix is too big? Cora is ~2700x2700 = 7M entries. It fits in memory.
    
    loss = norm * F.binary_cross_entropy_with_logits(reconstruction.view(-1), adj_label.view(-1), pos_weight=torch.tensor(pos_weight))
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            z = model(features, adj_norm, return_embeds=True)
            # We use CPU for evaluation to easily index with numpy edges
            emb = z.data.cpu().numpy()
            roc_curr, ap_curr = get_roc_score(emb, adj_orig, val_edges, val_edges_false)
            val_roc_history.append(roc_curr)
        
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "val_roc=", "{:.5f}".format(roc_curr), "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))
        loss_history.append(loss.item())

print("Optimization Finished!")

# Test
model.eval()
with torch.no_grad():
     z = model(features, adj_norm, return_embeds=True)
     emb = z.data.cpu().numpy()
     roc_score, ap_score = get_roc_score(emb, adj_orig, test_edges, test_edges_false)

print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))

# Save results
os.makedirs(args.save_dir, exist_ok=True)
np.save(os.path.join(args.save_dir, f"{args.dataset}_lp_loss.npy"), loss_history)
with open(os.path.join(args.save_dir, "results.txt"), "a") as f:
    f.write(f"{args.dataset} LP ROC: {roc_score} AP: {ap_score}\n")

