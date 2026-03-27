import os
import torch
import numpy as np
import scipy.sparse as sp
import warnings
from utils import load_graph_data, preprocess_graph, clustering, setup_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def run_fcm_on_facebook_datasets():
    # 1. Configuration
    # Facebook ego networks as defined in train.py
    # ego_ids = ["0", "107", "1684", "1912", "3437", "348", "3980", "414", "686", "698"]
    ego_ids = ["348", "414", "686", "698", "1684", "3980"]
    
    datasets = [f"facebook_{ego_id}" for ego_id in ego_ids]
    
    gnnlayers = 2  # Default for Facebook in train.py
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Running FCM on {len(datasets)} Facebook datasets...")
    print(f"Device: {device}")
    
    results = []

    # 2. Iterate through datasets
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")
        
        # Load Data
        try:
            # load_graph_data returns features (X), ground_truth_communities (y), and adjacency (A)
            # y is a list of sets (ground truth communities)
            X, y, A = load_graph_data(dataset_name, show_details=False)
            
            features = X
            true_labels = y  
            adj = sp.csr_matrix(A)
            
            # 3. Preprocessing (Laplacian Smoothing)
            # Correctly handle diagonal for preprocessing calculation
            adj_pre = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj_pre.eliminate_zeros()
            
            # Generate smoothing matrices
            adj_norm_s = preprocess_graph(adj_pre, gnnlayers, norm='sym', renorm=True)
            
            # Check for cached smoothed features (.npy) to save time
            path = f"dataset/facebook/{dataset_name}_feat_sm_{gnnlayers}.npy"
            sm_fea_s = None
            
            if os.path.exists(path):
                # print("  Loading smoothed features from disk...")
                sm_fea_s = np.load(path, allow_pickle=True)
            else:
                print("  Calculating and saving smoothed features...")
                sm_fea_s = sp.csr_matrix(features).toarray()
                for a in adj_norm_s:
                    sm_fea_s = a.dot(sm_fea_s)
                np.save(path, sm_fea_s, allow_pickle=True)

            # Convert to Tensor for clustering
            # Ensure it is float32 for skfuzzy/torch
            if isinstance(sm_fea_s, np.ndarray):
                sm_fea_s = torch.from_numpy(sm_fea_s).float().to(device)
            else:
                sm_fea_s = torch.FloatTensor(sm_fea_s).to(device)

            # 4. Setup Cluster Number (k)
            # Use ground truth K for baseline comparison
            cluster_num = len(true_labels)
            
            # 5. Run FCM Clustering
            setup_seed(42)  # Fix seed for reproducibility
            
            # clustering() in utils.py runs skfuzzy.cmeans and calculates metrics via eva()
            # It expects (feature, true_labels, cluster_num, device)
            # Returns: nmi, f1, acc, u, predict_labels_matrix, centers, dis
            nmi, f1, acc, _, _, _, _ = clustering(sm_fea_s, true_labels, cluster_num, device=device)
            
            print(f"  k={cluster_num} | NMI: {nmi:.2f} | F1: {f1:.2f} | Accuracy: {acc:.2f}")
            
            results.append({
                "dataset": dataset_name,
                "nmi": nmi,
                "f1": f1,
                "acc": acc,
                "k": cluster_num
            })
            
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # 6. Final Summary
    print("\n" + "="*65)
    print(f"{'Dataset':<20} {'k':<5} {'NMI':<10} {'F1':<10} {'ACC':<10}")
    print("-" * 65)
    
    avg_nmi = 0
    avg_f1 = 0
    avg_acc = 0
    
    for res in results:
        print(f"{res['dataset']:<20} {res['k']:<5} {res['nmi']:<10.2f} {res['f1']:<10.2f} {res['acc']:<10.2f}")
        avg_nmi += res['nmi']
        avg_f1 += res['f1']
        avg_acc += res['acc']
    
    if results:
        count = len(results)
        print("-" * 65)
        print(f"{'AVERAGE':<20} {'-':<5} {avg_nmi/count:<10.2f} {avg_f1/count:<10.2f} {avg_acc/count:<10.2f}")
        print("="*65)

if __name__ == "__main__":
    run_fcm_on_facebook_datasets()