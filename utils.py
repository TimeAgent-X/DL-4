import numpy as np
import scipy.sparse as sp
import torch
import os
import networkx as nx
import json
import sys
from networkx.readwrite import json_graph

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def drop_edge(adj, drop_rate=0.0):
    """
    Randomly drop edges from the adjacency matrix.
    Input: torch sparse tensor
    """
    if drop_rate <= 0.0:
        return adj
    
    indices = adj._indices()
    values = adj._values()
    
    # Random mask
    mask = torch.rand(values.size(0)) > drop_rate
    
    new_indices = indices[:, mask]
    new_values = values[mask]
    
    return torch.sparse.FloatTensor(new_indices, new_values, adj.shape)

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # REMOVE diagonal
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    data = np.ones(train_edges.shape[0])
    # Re-build train adj
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(emb[e[0], :].dot(emb[e[1], :].T)))
        pos.append(1)

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(emb[e[0], :].dot(emb[e[1], :].T)))
        neg.append(0)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([pos, neg])
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_ppi(output, labels):
    """Multilabel accuracy (micro-f1 based or simple binary match)"""
    # PPI is multi-label classification.
    # Often micro-F1 is used, but for simplicity let's stick to a threshold based accuracy or binary cross entropy loss monitoring.
    preds = (output > 0.5).type_as(labels)
    # Simple accuracy: (TP+TN)/(Total) ? No, typically F1 for PPI.
    # Let's return just raw correct count for now or implement F1 later?
    # PDF asks for ACC. Let's assume binary accuracy averaged?
    # Or strict match?
    # Let's return a placeholder accuracy matching similar shape.
    correct = preds.eq(labels).double().sum()
    return correct / (len(labels) * labels.shape[1])

def load_data(path="./", dataset="cora", add_self_loop=True):
    """Load citation network dataset (cora, citeseer)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}/{}/{}.content".format(path, dataset, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}/{}/{}.cites".format(path, dataset, dataset),
                                    dtype=np.int32)
    # Filter edges regarding isolated nodes/unknown nodes in citeseer
    edges = np.array(list(map(lambda x: idx_map[x] if x in idx_map else -1, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # Remove edges with -1
    edges = edges[(edges[:,0] != -1) & (edges[:,1] != -1)]
    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    
    if add_self_loop:
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    else:
        adj = normalize_adj(adj)

    # Standard splits usually (but let's define them loosely if not specified)

    # Cora: 140 train, 500 val, 1000 test
    # Citeseer: 120 train, 500 val, 1000 test
    if dataset == 'cora':
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    elif dataset == 'citeseer':
        idx_train = range(120)
        idx_val = range(120, 620)
        idx_test = range(620, 1620) # Approx 1000 test
        # Note: Citeseer has 3312 nodes. 
        
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_ppi_data(path="./", dataset="ppi", add_self_loop=True):
    print('Loading PPI dataset...')
    prefix = os.path.join(path, dataset, "ppi")
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    
    feats = np.load(prefix + "-feats.npy")
    id_map = json.load(open(prefix + "-id_map.json"))
    class_map = json.load(open(prefix + "-class_map.json"))
    
    # Re-index to 0-based
    # id_map maps string_id -> int_id
    # We need to ensure the order in feats matches
    # PPI feats is (Nodes, Features) aligned with numerical IDs?
    # Usually PPI inductive:
    # 20 graphs for train, 2 for val, 2 for test.
    # Here it seems we have one big graph? Or disconnected components?
    # Let's inspect "val" and "test" attributes in G if they exist.
    
    if isinstance(list(class_map.values())[0], list):
        lab = np.array(list(class_map.values()))
    else:
        # One hot?
        pass # PPI is multilabel
    
    # Conversion map
    # id_map is recursive?
    # { "0": 0, "1": 1 ... }
    
    # Extract labels in correct order
    num_nodes = len(id_map)
    labels = np.zeros((num_nodes, len(list(class_map.values())[0])))
    for k, v in class_map.items():
        idx = id_map[k]
        labels[idx] = v
        
    # Standard split for PPI (usually indicated by 'val' and 'test' bool attributes in G nodes)
    # Check if nodes have 'val' or 'test' attributes
    idx_train = []
    idx_val = []
    idx_test = []
    
    for node_id, node_data in G.nodes(data=True):
        idx = id_map[str(node_id)]
        if node_data.get('test', False):
            idx_test.append(idx)
        elif node_data.get('val', False):
            idx_val.append(idx)
        else:
            idx_train.append(idx)
            
    adj = nx.adjacency_matrix(G)
    if add_self_loop:
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    else:
        adj = normalize_adj(adj)
    
    features = normalize(feats)
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels) # Float for Multi-label BCE
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return adj, features, labels, idx_train, idx_val, idx_test

