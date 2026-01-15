from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, load_ppi_data, accuracy, accuracy_ppi, drop_edge
from model import GCN, GCN_PPI

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
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    help='Dataset to use (cora, citeseer, ppi).')
parser.add_argument('--layers', type=int, default=2,
                    help='Number of GCN layers.')
parser.add_argument('--no_self_loop', action='store_true', default=False,
                    help='Disable self loops.')
parser.add_argument('--drop_edge', type=float, default=0.0,
                    help='DropEdge rate.')
parser.add_argument('--pairnorm', action='store_true', default=False,
                    help='Use PairNorm.')
parser.add_argument('--save_dir', type=str, default="plots",
                    help='Directory to save plots/logs.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.dataset == "ppi":
    adj, features, labels, idx_train, idx_val, idx_test = load_ppi_data(dataset=args.dataset, add_self_loop=not args.no_self_loop)
    ModelClass = GCN_PPI
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    accuracy_fcn = accuracy_ppi
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset, add_self_loop=not args.no_self_loop)
    ModelClass = GCN
    loss_fcn = F.nll_loss
    accuracy_fcn = accuracy

# Model and optimizer
model = ModelClass(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.shape[1] if args.dataset == "ppi" else labels.max().item() + 1,
            dropout=args.dropout,
            n_layers=args.layers,
            use_pairnorm=args.pairnorm)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# Trackers
loss_train_hist = []
loss_val_hist = []
acc_val_hist = []

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    # DropEdge
    adj_train = adj
    if args.drop_edge > 0:
        adj_train = drop_edge(adj, args.drop_edge)
    
    output = model(features, adj_train)
    
    if args.dataset == "ppi":
        loss_train = loss_fcn(output[idx_train], labels[idx_train])
        acc_train = accuracy_fcn(output[idx_train], labels[idx_train])
    else:
        loss_train = loss_fcn(output[idx_train], labels[idx_train])
        acc_train = accuracy_fcn(output[idx_train], labels[idx_train])
        
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    if args.dataset == "ppi":
         loss_val = loss_fcn(output[idx_val], labels[idx_val])
         acc_val = accuracy_fcn(output[idx_val], labels[idx_val])
    else:
         loss_val = loss_fcn(output[idx_val], labels[idx_val])
         acc_val = accuracy_fcn(output[idx_val], labels[idx_val])
    
    loss_train_hist.append(loss_train.item())
    loss_val_hist.append(loss_val.item())
    acc_val_hist.append(acc_val.item())

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    if args.dataset == "ppi":
        loss_test = loss_fcn(output[idx_test], labels[idx_test])
        acc_test = accuracy_fcn(output[idx_test], labels[idx_test])
    else:
        loss_test = loss_fcn(output[idx_test], labels[idx_test])
        acc_test = accuracy_fcn(output[idx_test], labels[idx_test])
        
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    
    return acc_test.item()


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test_acc = test()

# Save results to file for plotting later if needed
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

filename_prefix = "{}_l{}_n{}_d{}_de{}".format(args.dataset, args.layers, int(args.pairnorm), args.dropout, args.drop_edge)
if args.no_self_loop:
    filename_prefix += "_noself"

np.save(os.path.join(args.save_dir, filename_prefix + "_loss_train.npy"), loss_train_hist)
np.save(os.path.join(args.save_dir, filename_prefix + "_loss_val.npy"), loss_val_hist)

# Write results
with open(os.path.join(args.save_dir, "results.txt"), "a") as f:
    f.write(f"{filename_prefix} Test Acc: {test_acc}\n")

