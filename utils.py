import os
from collections import defaultdict
import torch
import random
import math
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
from kmeans_gpu import kmeans                              
import skfuzzy as fuzz
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


# def eva(y_true, y_pred, show_details=True):
#     """
#     evaluate the clustering performance
#     Args:
#         y_true: the ground truth
#         y_pred: the predicted label
#         show_details: if print the details
#     Returns: None
#     """
#     nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
#     ari = ari_score(y_true, y_pred)

#     return nmi, ari

def eva(C_star, C_hat, num_nodes):
    """
    Calculates ONMI, Average F1 Score, and Pairwise Accuracy 
    for overlapping communities.
    """
    # ==========================================
    # Helper Functions
    # ==========================================
    def entropy(p_list):
        return -sum([p * math.log2(p) for p in p_list if p > 0])

    def calc_conditional_entropy(c1, c2, N):
        intersect = len(c1.intersection(c2))
        c1_only = len(c1) - intersect
        c2_only = len(c2) - intersect
        neither = N - (intersect + c1_only + c2_only)
        
        p_11 = intersect / N
        p_10 = c1_only / N
        p_01 = c2_only / N
        p_00 = neither / N
        
        p_x1 = len(c1) / N
        p_x0 = 1.0 - p_x1
        p_y1 = len(c2) / N
        p_y0 = 1.0 - p_y1
        
        H_X = entropy([p_x0, p_x1])
        H_Y = entropy([p_y0, p_y1])
        H_XY = entropy([p_00, p_01, p_10, p_11])
        
        H_X_given_Y = H_XY - H_Y
        return H_X, H_Y, H_X_given_Y

    def calc_f1(set_a, set_b):
        """Helper to calculate F1 score between two sets of nodes."""
        if not set_a or not set_b:
            return 0.0
        intersect = len(set_a.intersection(set_b))
        return (2.0 * intersect) / (len(set_a) + len(set_b))

    # Base check
    if not C_star or not C_hat:
        return 0.0, 0.0, 0.0

    # ==========================================
    # 1. ONMI Calculation
    # ==========================================
    H_X_given_Y_norm = []
    for c_x in C_star:
        H_X, _, _ = calc_conditional_entropy(c_x, set(), num_nodes)
        if H_X == 0:
            continue
            
        min_H_X_given_Y = float('inf')
        for c_y in C_hat:
            _, _, H_X_given_Y = calc_conditional_entropy(c_x, c_y, num_nodes)
            min_H_X_given_Y = min(min_H_X_given_Y, H_X_given_Y)
            
        H_X_given_Y_norm.append(min(1.0, min_H_X_given_Y / H_X))

    H_Y_given_X_norm = []
    for c_y in C_hat:
        H_Y, _, _ = calc_conditional_entropy(c_y, set(), num_nodes)
        if H_Y == 0:
            continue
            
        min_H_Y_given_X = float('inf')
        for c_x in C_star:
            _, _, H_Y_given_X = calc_conditional_entropy(c_y, c_x, num_nodes)
            min_H_Y_given_X = min(min_H_Y_given_X, H_Y_given_X)
            
        H_Y_given_X_norm.append(min(1.0, min_H_Y_given_X / H_Y))

    avg_H_X_given_Y = np.mean(H_X_given_Y_norm) if H_X_given_Y_norm else 1.0
    avg_H_Y_given_X = np.mean(H_Y_given_X_norm) if H_Y_given_X_norm else 1.0
    
    onmi = max(0.0, 1.0 - 0.5 * (avg_H_X_given_Y + avg_H_Y_given_X))

    # ==========================================
    # 2. Average F1 Score Calculation
    # ==========================================
    f1_star_to_hat = []
    for c_x in C_star:
        best_f1 = max([calc_f1(c_x, c_y) for c_y in C_hat])
        f1_star_to_hat.append(best_f1)
        
    f1_hat_to_star = []
    for c_y in C_hat:
        best_f1 = max([calc_f1(c_y, c_x) for c_x in C_star])
        f1_hat_to_star.append(best_f1)
        
    avg_f1 = 0.5 * (np.mean(f1_star_to_hat) + np.mean(f1_hat_to_star))

    # ==========================================
    # 3. Pairwise Accuracy Calculation
    # ==========================================
    shared_true = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    shared_pred = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    
    for c in C_star:
        nodes = list(c)
        for i in range(len(nodes)):
            for j in range(i, len(nodes)): 
                shared_true[nodes[i], nodes[j]] += 1
                if i != j: shared_true[nodes[j], nodes[i]] += 1
                
    for c in C_hat:
        nodes = list(c)
        for i in range(len(nodes)):
            for j in range(i, len(nodes)):
                shared_pred[nodes[i], nodes[j]] += 1
                if i != j: shared_pred[nodes[j], nodes[i]] += 1
                
    # Fraction of pairs where the number of shared communities matches exactly
    accuracy = np.mean(shared_true == shared_pred)

    return onmi, avg_f1, accuracy

# def eva(C_star, C_hat, num_nodes):
#     """
#     Evaluates detected communities against ground truth using BigClam metrics.
    
#     Parameters
#     ----------
#     C_star : list of set of int
#         Ground truth communities (each set contains node indices).
#     C_hat : list of set of int
#         Detected communities (each set contains node indices).
#     num_nodes : int
#         Total number of nodes in the graph.
        
#     Returns
#     -------
#     dict
#         Dictionary containing Average F1, Omega Index, NMI, and Number Accuracy.
#     """
    
#     def calc_f1(set_a, set_b):
#         """Helper to calculate F1 score between two sets of nodes."""
#         if not set_a or not set_b:
#             return 0.0
#         intersect = len(set_a.intersection(set_b))
#         return (2.0 * intersect) / (len(set_a) + len(set_b))

#     # 1. Average F1 Score
#     # Defined as the average of the F1-score of the best-matching ground-truth 
#     # community to each detected community, and vice versa.
#     f1_star_to_hat = []
#     for c_i in C_star:
#         best_f1 = max([calc_f1(c_i, c_j) for c_j in C_hat]) if C_hat else 0.0
#         f1_star_to_hat.append(best_f1)
        
#     f1_hat_to_star = []
#     for c_j in C_hat:
#         best_f1 = max([calc_f1(c_j, c_i) for c_i in C_star]) if C_star else 0.0
#         f1_hat_to_star.append(best_f1)
        
#     avg_f1_star = np.mean(f1_star_to_hat) if f1_star_to_hat else 0.0
#     avg_f1_hat = np.mean(f1_hat_to_star) if f1_hat_to_star else 0.0
#     avg_f1 = 0.5 * (avg_f1_star + avg_f1_hat) # [cite: 404]

#     # 2. Omega Index
#     # Accuracy on estimating the number of communities that each pair of nodes shares.
#     shared_true = np.zeros((num_nodes, num_nodes), dtype=np.int32)
#     shared_pred = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    
#     for c in C_star:
#         nodes = list(c)
#         for i in range(len(nodes)):
#             for j in range(i, len(nodes)): 
#                 shared_true[nodes[i], nodes[j]] += 1
#                 if i != j: shared_true[nodes[j], nodes[i]] += 1
                
#     for c in C_hat:
#         nodes = list(c)
#         for i in range(len(nodes)):
#             for j in range(i, len(nodes)):
#                 shared_pred[nodes[i], nodes[j]] += 1
#                 if i != j: shared_pred[nodes[j], nodes[i]] += 1
                
#     # Fraction of pairs (u, v) where the number of shared communities is exactly the same
#     omega_index = np.mean(shared_true == shared_pred) # [cite: 407, 408]

#     # 3. Normalized Mutual Information (NMI)
#     # For overlapping communities, a common approximation is flattening the binary 
#     # membership matrices, as standard NMI expects mutually exclusive labels.
#     Y_true_bin = np.zeros((num_nodes, len(C_star)), dtype=np.int32)
#     for i, c in enumerate(C_star):
#         for node in c: Y_true_bin[node, i] = 1
            
#     Y_pred_bin = np.zeros((num_nodes, len(C_hat)), dtype=np.int32)
#     for i, c in enumerate(C_hat):
#         for node in c: Y_pred_bin[node, i] = 1

#     # Flatten arrays to compute standard NMI on overlapping matrices
#     nmi_score = nmi_score(Y_true_bin.flatten(), Y_pred_bin.flatten()) # [cite: 410, 411]

#     # 4. Accuracy in the number of communities
#     # Relative accuracy between the detected and the true number of communities.
#     num_true = len(C_star)
#     num_pred = len(C_hat)
#     if num_true == 0:
#         num_acc = 0.0
#     else:
#         # Penalizes over-prediction and under-prediction relative to ground truth
#         num_acc = max(0.0, 1.0 - abs(num_true - num_pred) / (2.0 * num_true)) # [cite: 412, 413]

#     return {
#         "Average_F1": avg_f1,
#         "Omega_Index": omega_index,
#         "NMI": nmi_score,
#         "Num_Communities_Accuracy": num_acc
#     }

def _load_facebook_data(dataset_dir, ego_id):
    """Load a graph from raw facebook files in the dataset directory."""
    feat_path = os.path.join(dataset_dir, f"{ego_id}.feat")
    egofeat_path = os.path.join(dataset_dir, f"{ego_id}.egofeat")
    edge_path = os.path.join(dataset_dir, f"{ego_id}.edges")
    circle_path = os.path.join(dataset_dir, f"{ego_id}.circles")

    id_map = {}
    all_features = []

    # 1. Load Alter Nodes Features
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                node_id = int(parts[0])
                feats = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                if node_id not in id_map:
                    id_map[node_id] = len(id_map)
                    all_features.append(feats)

    if not all_features:
        if os.path.exists(egofeat_path):
             pass # Add ego features here if desired
        if not all_features:
             raise ValueError(f"No feature data found for ego {ego_id}")
    
    X = np.array(all_features)
    num_nodes = len(id_map)

    # 2. Build Adjacency Matrix
    adj_set = set()
    if os.path.exists(edge_path):
        with open(edge_path, 'r') as f:
            for line in f:
                u, v = map(int, line.strip().split())
                if u in id_map and v in id_map:
                    u_idx, v_idx = id_map[u], id_map[v]
                    adj_set.add((u_idx, v_idx))
                    adj_set.add((v_idx, u_idx))

    rows, cols = zip(*adj_set) if adj_set else ([], [])
    data = np.ones(len(rows)) 
    A = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    # 3. Labels (Circles) - GROUND TRUTH FORMAT (List of Sets)
    ground_truth_communities = []
    
    if os.path.exists(circle_path):
        with open(circle_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Format: circle<id> node1 node2 ...
                circle_nodes = set()
                for node_str in parts[1:]:
                    node_id = int(node_str)
                    if node_id in id_map:
                        circle_nodes.add(id_map[node_id])
                
                if circle_nodes:
                    ground_truth_communities.append(circle_nodes)

    return X, ground_truth_communities, A

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

# def clustering(feature, true_labels, cluster_num, device):
#     # Ensure feature is on the correct device
#     feature = feature.to(device)

#     # 1. Instantiate FCM
#     model = FCM(
#         num_features=feature.shape[1], 
#         num_clusters=cluster_num, 
#         device=device
#     ).to(device)

#     # 2. Fit FCM (Iterative Update)
#     model.fit(feature)

#     # 3. Get Overlapping Predictions
#     # threshold=0.25 allows multiple clusters per node if their membership is strong enough
#     predict_labels_matrix, centers, dis = model.predict(feature, threshold=0.25)

#     # 4. Evaluation for logging 
#     # (Collapsing overlapping labels to single dominant label for NMI/ARI scoring)
#     dominant_labels = torch.argmax(predict_labels_matrix, dim=1).numpy()
#     nmi, ari = eva(true_labels, dominant_labels, show_details=False)
    
#     # 5. Return the full OVERLAPPING matrix for the RL agent
#     return 100 * nmi, 100 * ari, predict_labels_matrix, centers, dis

def clustering(feature, true_labels, cluster_num, device):
    # 1. Prepare data for fuzzy c-means (requires CPU numpy array)
    # feature shape: (N, D) -> skfuzzy requires (D, N)
    feature_np = feature.detach().cpu().numpy().T  
    
    # 2. Run Fuzzy C-Means
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data=feature_np, 
        c=cluster_num, 
        m=2.0, 
        error=0.005, 
        maxiter=1000, 
        init=None
    )

    u = u.T  # Transpose back to (n_samples, n_clusters)

    # 3. Move results back to PyTorch and correct device
    centers = torch.from_numpy(cntr).float().to(device)
    predict_labels_soft = torch.from_numpy(u).float().to(device)

    '''    
    # Convert U to hard labels using argmax for dominant cluster assignment
    predict_labels_hard = torch.argmax(predict_labels_soft, dim=1)

    # Convert predict_labels_hard into a list of sets format (C_hat)
    predict_labels_hard_np = predict_labels_hard.cpu().numpy()
    C_hat = []
    for c in range(cluster_num):
        member_nodes = set(np.where(predict_labels_hard_np == c)[0].tolist())
        C_hat.append(member_nodes)
    '''

    # 4. Process labels
    u_max = predict_labels_soft.max(dim=1, keepdim=True).values
    u_norm = predict_labels_soft / (u_max + 1e-8)

    predict_labels_matrix = (u_norm > 0.5).float()
    
    # Convert the (N, K) binary matrix into a list of sets format (C_hat)
    predict_labels_matrix_np = predict_labels_matrix.cpu().numpy()
    C_hat = []
    
    for c in range(cluster_num):
        # Find all row indices (nodes) where column 'c' is 1
        member_nodes = set(np.where(predict_labels_matrix_np[:, c] == 1.0)[0].tolist())
        C_hat.append(member_nodes)
        
    # Calculate NMI using the custom BigClam evaluation function true_labels must be passed in as C_star (list of sets)
    nmi, f1, acc = eva(true_labels, C_hat, num_nodes=feature.shape[0])

    # 5. Compute distances
    # Ensure 'feature' is on the same device as 'centers'
    dis = torch.cdist(feature.to(device), centers).pow(2)
    
    return 100 * nmi, 100 * f1, 100 * acc, predict_labels_soft, predict_labels_matrix, centers, dis
    
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
