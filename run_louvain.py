"""
Execution script to evaluate Louvain community detection on Facebook datasets.
Provides baseline overlapping NMI and F1 comparisons without deep learning.
"""
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import scipy.sparse as sp
import warnings
from utils import load_graph_data, eva, setup_seed

warnings.filterwarnings("ignore")

def run_louvain_on_facebook_datasets():
    ego_ids = ["107","348", "414", "686", "698"]
    datasets = [f"facebook_{ego_id}" for ego_id in ego_ids]
    
    print(f"Running Louvain on {len(datasets)} Facebook datasets...")

    summary_results = []

    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")
        
        k_list = []
        nmi_list = []
        f1_list = []
        
        try:
            _, true_labels, adj = load_graph_data(dataset_name, show_details=False)
            
            if not sp.issparse(adj):
                adj = sp.csr_matrix(adj)
            
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()

            if hasattr(nx, 'from_scipy_sparse_array'):
                G = nx.from_scipy_sparse_array(adj)
            else:
                G = nx.from_scipy_sparse_matrix(adj)
            
            for seed in range(10):
                setup_seed(seed)
                
                if hasattr(nx_comm, 'louvain_communities'):
                    communities = nx_comm.louvain_communities(G, seed=seed)
                else:
                    import community as community_louvain
                    partition = community_louvain.best_partition(G, random_state=seed)
                    communities = [set() for _ in range(len(set(partition.values())))]
                    for node, comm_id in partition.items():
                        communities[comm_id].add(node)
                
                nmi, f1 = eva(true_labels, list(communities))
                
                k_list.append(len(communities))
                nmi_list.append(nmi * 100)
                f1_list.append(f1 * 100)

            k_list = np.array(k_list)
            nmi_list = np.array(nmi_list)
            f1_list = np.array(f1_list)
            
            k_mean = k_list.mean()
            k_list_std = k_list.std()
            nmi_mean = nmi_list.mean()
            nmi_std = nmi_list.std()
            f1_mean = f1_list.mean()
            f1_std = f1_list.std()

            print(f"Results for {dataset_name}:")
            print(f"k:   {k_mean:.1f} ± {k_list_std:.1f}")
            print(f"NMI: {nmi_mean:.4f} ± {nmi_std:.4f}")
            print(f"F1:  {f1_mean:.4f} ± {f1_std:.4f}")
            print("-" * 20)
            
            summary_results.append({
                "dataset": dataset_name,
                "k_mean": k_mean,
                "k_std": k_list_std,
                "nmi_mean": nmi_mean,
                "nmi_std": nmi_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std
            })

        except Exception as e:
            print(f"Error on {dataset_name}: {e}")

    if summary_results:
        print("\n" + "="*80)
        print(f"{'Dataset':<20} {'k (Mean ± Std)':<22} {'NMI (Mean ± Std)':<22} {'F1 (Mean ± Std)':<22}")
        print("-" * 80)
        
        avg_nmi_mean = 0
        avg_f1_mean = 0
        
        for res in summary_results:
            k_str = f"{res['k_mean']:.1f} ± {res['k_std']:.1f}"
            nmi_str = f"{res['nmi_mean']:.2f} ± {res['nmi_std']:.2f}"
            f1_str = f"{res['f1_mean']:.2f} ± {res['f1_std']:.2f}"
            
            print(f"{res['dataset']:<20} {k_str:<22} {nmi_str:<22} {f1_str:<22}")
            
            avg_nmi_mean += res['nmi_mean']
            avg_f1_mean += res['f1_mean']
        
        print("-" * 80)
        count = len(summary_results)
        print(f"{'OVERALL AVERAGE':<20} {'-':<10} {avg_nmi_mean/count:<22.2f} {avg_f1_mean/count:<22.2f}")
        print("="*80)

if __name__ == "__main__":
    run_louvain_on_facebook_datasets()