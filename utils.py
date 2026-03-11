import os
from collections import defaultdict
import torch
import random
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
from kmeans_gpu import kmeans                              
from model import CAN         
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [1] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))

    return adjs


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # ss = silhouette_score(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))

    return nmi, ari

def _load_facebook_data(dataset_dir, ego_id):
    """Load a graph from raw facebook files in the dataset directory.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory (e.g., 'dataset/facebook').
    ego_id : str
        The ego node ID (e.g., '0').

    Returns
    -------
    feat, label, adj
    """
    feat_path = os.path.join(dataset_dir, f"{ego_id}.feat")
    egofeat_path = os.path.join(dataset_dir, f"{ego_id}.egofeat")
    edge_path = os.path.join(dataset_dir, f"{ego_id}.edges")
    circle_path = os.path.join(dataset_dir, f"{ego_id}.circles")

    # Map external node ID to internal index
    id_map = {}
    all_features = []

    # 1. Load Ego Node Feature
    ego_val = int(ego_id)
    if os.path.exists(egofeat_path):
        with open(egofeat_path, 'r') as f:
            content = f.read().strip().split()
            ego_features = np.array([float(x) for x in content], dtype=np.float32)
            
            id_map[ego_val] = 0
            all_features.append(ego_features)

    # 2. Load Alter Nodes Features
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                node_id = int(parts[0])
                feats = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                if node_id not in id_map:
                    id_map[node_id] = len(id_map)
                    all_features.append(feats)
    
    # If no features found, create identity or handle gracefully? 
    # Usually SNAP data has features.
    if len(all_features) == 0:
        # Fallback if no features - just identity for ego
        # But this likely means paths are wrong
        print(f"Warning: No features found for ego {ego_id} in {dataset_dir}")
        X = np.eye(1, dtype=np.float32)
    else:
        X = np.array(all_features)
    
    num_nodes = len(id_map)

    # 3. Edges
    edges = []
    if os.path.exists(edge_path):
        with open(edge_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    if u in id_map and v in id_map:
                        edges.append((id_map[u], id_map[v]))
                        edges.append((id_map[v], id_map[u]))

    # Add edges from Ego to everyone else
    if ego_val in id_map:
        ego_idx = id_map[ego_val]
        for node_id, idx in id_map.items():
            if idx != ego_idx:
                edges.append((ego_idx, idx))
                edges.append((idx, ego_idx))

    if len(edges) > 0:
        edges = np.array(edges)
        row = edges[:, 0]
        col = edges[:, 1]
        data = np.ones(len(edges))
        A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    else:
        A = sp.eye(num_nodes, format='csr')

    # 4. Labels (Circles)
    # Default label -1 or 0
    labels = np.zeros(num_nodes, dtype=np.int32)
    
    if os.path.exists(circle_path):
        with open(circle_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                # format: circle<id> node1 node2 ...
                for node_str in parts[1:]:
                    node_id = int(node_str)
                    if node_id in id_map:
                        labels[id_map[node_id]] = i

    return X, labels, A

def load_graph_data(dataset_name, show_details=False):
    """
    load graph dataset
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj
    """
    if dataset_name.startswith("facebook"):
        folder = "dataset/facebook"
        if "_" in dataset_name:
            ego_id = dataset_name.split("_")[1]
        else:
            ego_id = "3980"
        return _load_facebook_data(folder, ego_id)
    
    load_path = "dataset/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    return feat, label, adj


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# def clustering(feature, true_labels, cluster_num, device):
#     predict_labels, centers, dis = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device=device)

#     nmi, ari = eva(true_labels, predict_labels.numpy(), show_details=False)
#     return 100 * nmi, 100 * ari, predict_labels.numpy(), centers, dis

def clustering(feature, true_labels, cluster_num, device):
    # Ensure feature is on the correct device
    feature = feature.to(device)

    # 1. Instantiate the CAN model for the current batch
    model = CAN(
        num_nodes=feature.shape[0], 
        num_features=feature.shape[1], 
        num_clusters=cluster_num, 
        device=device
    ).to(device)

    # 2. Train (fit) the model on the current embeddings
    # We pass the feature to fit, which returns the shift value used
    X_min = model.fit(feature)

    # 3. Get predictions
    predict_labels_matrix, centers, dis = model.predict(feature, X_min, threshold=0.5)

    # 4. Evaluation for logging (Collapse overlapping labels to single dominant label)
    # We use argmax to force a single label for NMI/ARI calculation
    dominant_labels = torch.argmax(predict_labels_matrix, dim=1).numpy()
    nmi, ari = eva(true_labels, dominant_labels, show_details=False)
    
    # 5. Return the full overlapping matrix for the RL agent
    return 100 * nmi, 100 * ari, predict_labels_matrix, centers, dis


def diffusion_adj(adj, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self_loop
    adj_tmp = adj + np.eye(adj.shape[0])

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    sqrt_d_inv = np.sqrt(d_inv)

    # calculate norm adj
    norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # calculate graph diffusion
    if mode == "ppr":
        diff_adj = transport_rate * np.linalg.inv((np.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))

    return diff_adj


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj
