import os
import torch
import random
import numpy as np
import scipy.sparse as sp
import skfuzzy as fuzz
from cdlib import NodeClustering, evaluation


def preprocess_graph(adj, layer, norm='sym', renorm=True):
    """
    Standardize/Normalize adjcency matrix inputs iteratively.
    Adds a self-loop (renorm option) and applies matrix normalization.
    """
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


def eva(C_star, C_hat, num_nodes=None):
    """
    Calculates ONMI and Average F1 Score for overlapping communities 
    using cdlib's implementation.
    """
    # Base check
    if not C_star or not C_hat:
        return 0.0, 0.0

    nc_star = NodeClustering(list(C_star), graph=None, method_name="ground_truth")
    nc_hat = NodeClustering(list(C_hat), graph=None, method_name="prediction")

    onmi_result = evaluation.overlapping_normalized_mutual_information_MGH(nc_star, nc_hat)
    onmi = onmi_result.score

    f1_result = evaluation.f1(nc_star, nc_hat)
    avg_f1 = f1_result.score

    return onmi, avg_f1

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


def clustering(feature, true_labels, cluster_num, device, threshold=0.5):
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

    # 4. Process labels
    u_max = predict_labels_soft.max(dim=1, keepdim=True).values
    u_norm = predict_labels_soft / (u_max + 1e-8)

    # Use the parameter threshold instead of hardcoded 0.5
    predict_labels_matrix = (u_norm > threshold).float()
    
    # Convert the (N, K) binary matrix into a list of sets format (C_hat)
    predict_labels_matrix_np = predict_labels_matrix.cpu().numpy()
    C_hat = []
    
    for c in range(cluster_num):
        # Find all row indices (nodes) where column 'c' is 1
        member_nodes = set(np.where(predict_labels_matrix_np[:, c] == 1.0)[0].tolist())
        
        # FILTER: Ignore empty ghost clusters and exact duplicate clone clusters
        if len(member_nodes) > 0 and member_nodes not in C_hat:
            C_hat.append(member_nodes)
        
    # Calculate NMI using the custom BigClam evaluation function true_labels must be passed in as C_star (list of sets)
    nmi, f1 = eva(true_labels, C_hat, num_nodes=feature.shape[0])

    # 5. Compute distances
    # Ensure 'feature' is on the same device as 'centers'
    dis = torch.cdist(feature.to(device), centers).pow(2)
    
    return 100 * nmi, 100 * f1, predict_labels_soft, predict_labels_matrix, centers, dis
    
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
