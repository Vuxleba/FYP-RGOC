"""
Execution script to evaluate standard Fuzzy C-Means (FCM) clustering
accuracy on the pre-processed Facebook graphs.
"""
import torch
import numpy as np
import scipy.sparse as sp
import warnings
from utils import load_graph_data, clustering, setup_seed, preprocess_graph
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# Hyperparameters meticulously matched to the dataset routing in train.py
HYPERPARAMS = {
    "facebook_107":  {"n_input": -1,  "threshold": 0.95, "gnnlayers": 2},
    "facebook_348":  {"n_input": 128, "threshold": 0.8,  "gnnlayers": 5},
    "facebook_414":  {"n_input": -1,  "threshold": 0.8,  "gnnlayers": 3},
    "facebook_686":  {"n_input": -1,  "threshold": 0.7,  "gnnlayers": 1},
    "facebook_698":  {"n_input": -1,  "threshold": 0.9,  "gnnlayers": 2},
}

def run_fcm_on_facebook_datasets():
    datasets = list(HYPERPARAMS.keys())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Running FCM on {len(datasets)} Facebook datasets...")
    print(f"Device: {device}")

    summary_results = []

    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")
        
        nmi_list = []
        f1_list = []
        
        # Fetch dataset-specific params
        n_input = HYPERPARAMS[dataset_name]["n_input"]
        threshold = HYPERPARAMS[dataset_name]["threshold"]
        gnnlayers = HYPERPARAMS[dataset_name]["gnnlayers"]
        
        try:
            X, y, A = load_graph_data(dataset_name, show_details=False)
            features = X
            adj = sp.csr_matrix(A)
            
            if n_input != -1:
                pca = PCA(n_components=n_input)
                features = pca.fit_transform(features)
                
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()
            
            adj_norm_s = preprocess_graph(adj, gnnlayers, norm='sym', renorm=True)
            sm_fea_s = sp.csr_matrix(features).toarray()
            
            for a in adj_norm_s:
                sm_fea_s = a.dot(sm_fea_s)

            features_tensor = torch.FloatTensor(sm_fea_s).to(device)
            cluster_num = len(y)
            
            for seed in range(10):
                setup_seed(seed)  
                
                nmi, f1, predict_labels_soft, predict_labels_matrix, centers, dis = clustering(
                    features_tensor, 
                    y, 
                    cluster_num, 
                    device=device, 
                    threshold=threshold
                )
                
                nmi_list.append(nmi)
                f1_list.append(f1)
            
            nmi_list = np.array(nmi_list)
            f1_list = np.array(f1_list)
            
            nmi_mean, nmi_std = nmi_list.mean(), nmi_list.std()
            f1_mean, f1_std = f1_list.mean(), f1_list.std()

            print(f"Results for {dataset_name} (k={cluster_num}, PCA={n_input}, Thresh={threshold}, Layers={gnnlayers}):")
            print(f"NMI: {nmi_mean:.4f} ± {nmi_std:.4f}")
            print(f"F1:  {f1_mean:.4f} ± {f1_std:.4f}")
            print("-" * 20)
            
            summary_results.append({
                "dataset": dataset_name,
                "k": cluster_num,
                "nmi_mean": nmi_mean,
                "nmi_std": nmi_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std
            })
            
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            print("-" * 20)

    if summary_results:
        print("\n" + "="*75)
        print(f"{'Dataset':<20} {'k':<5} {'NMI (Mean ± Std)':<22} {'F1 (Mean ± Std)':<22}")
        print("-" * 75)
        
        avg_nmi_mean = 0
        avg_f1_mean = 0
        
        for res in summary_results:
            nmi_str = f"{res['nmi_mean']:.2f} ± {res['nmi_std']:.2f}"
            f1_str = f"{res['f1_mean']:.2f} ± {res['f1_std']:.2f}"
            
            print(f"{res['dataset']:<20} {res['k']:<5} {nmi_str:<22} {f1_str:<22}")
            
            avg_nmi_mean += res['nmi_mean']
            avg_f1_mean += res['f1_mean']
        
        print("-" * 75)
        count = len(summary_results)
        print(f"{'OVERALL AVERAGE':<20} {'-':<5} {avg_nmi_mean/count:<22.2f} {avg_f1_mean/count:<22.2f}")
        print("="*75)

if __name__ == "__main__":
    run_fcm_on_facebook_datasets()