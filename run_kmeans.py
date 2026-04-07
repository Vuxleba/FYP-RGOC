"""
Execution script to evaluate standard K-Means clustering.
Serves as an additional performance baseline.
"""
import numpy as np
import scipy.sparse as sp
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils import load_graph_data, setup_seed, preprocess_graph, eva

warnings.filterwarnings("ignore")

HYPERPARAMS = {
    "facebook_107":  {"n_input": -1,  "gnnlayers": 2},
    "facebook_348":  {"n_input": 128, "gnnlayers": 5},
    "facebook_414":  {"n_input": -1,  "gnnlayers": 3},
    "facebook_686":  {"n_input": -1,  "gnnlayers": 1},
    "facebook_698":  {"n_input": -1,  "gnnlayers": 2},
}

def run_kmeans_on_facebook_datasets():
    datasets = list(HYPERPARAMS.keys())
    
    print(f"Running Smoothed K-Means (Matched to train.py) on {len(datasets)} Facebook datasets...")

    summary_results = []

    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")
        
        nmi_list = []
        f1_list = []
        
        n_input = HYPERPARAMS[dataset_name]["n_input"]
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

            cluster_num = len(y) 
            
            for seed in range(10):
                setup_seed(seed)  
                
                kmeans = KMeans(n_clusters=cluster_num, random_state=seed, n_init=10)
                labels = kmeans.fit_predict(sm_fea_s)
                
                C_hat = []
                for c in range(cluster_num):
                    member_nodes = set(np.where(labels == c)[0].tolist())
                    if len(member_nodes) > 0:
                        C_hat.append(member_nodes)
                
                nmi, f1 = eva(y, C_hat, num_nodes=sm_fea_s.shape[0])
                
                nmi_list.append(nmi * 100)
                f1_list.append(f1 * 100)
            
            nmi_list = np.array(nmi_list)
            f1_list = np.array(f1_list)
            
            nmi_mean, nmi_std = nmi_list.mean(), nmi_list.std()
            f1_mean, f1_std = f1_list.mean(), f1_list.std()

            print(f"Results for {dataset_name} (k={cluster_num}, PCA={n_input}, Layers={gnnlayers}):")
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
        print("\n" + "="*70)
        print(f"{'Dataset':<20} {'k':<5} {'NMI (Mean ± Std)':<20} {'F1 (Mean ± Std)':<20}")
        print("-" * 70)
        
        avg_nmi_mean = 0
        avg_f1_mean = 0
        
        for res in summary_results:
            nmi_str = f"{res['nmi_mean']:.2f} ± {res['nmi_std']:.2f}"
            f1_str = f"{res['f1_mean']:.2f} ± {res['f1_std']:.2f}"
            
            print(f"{res['dataset']:<20} {res['k']:<5} {nmi_str:<20} {f1_str:<20}")
            
            avg_nmi_mean += res['nmi_mean']
            avg_f1_mean += res['f1_mean']
        
        print("-" * 70)
        count = len(summary_results)
        print(f"{'OVERALL AVERAGE':<20} {'-':<5} {avg_nmi_mean/count:<20.2f} {avg_f1_mean/count:<20.2f}")
        print("="*70)

if __name__ == "__main__":
    run_kmeans_on_facebook_datasets()