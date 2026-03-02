import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import scipy.sparse as sp
import warnings
from utils import load_graph_data, eva, setup_seed

warnings.filterwarnings("ignore")

# Define the datasets to process
datasets = ["bat", "eat", "cora", "citeseer"]

print(f"Running Louvain on: {datasets}")

for dataset in datasets:
    print(f"Processing {dataset}...")
    
    k_list = []
    modularity_list = []
    nmi_list = []
    ari_list = []

    for seed in range(10):
        # Set seed for reproducibility
        setup_seed(seed)
        
        try:
            # Load data
            _, true_labels, adj = load_graph_data(dataset, show_details=False)
            
            # Process Adjacency Matrix
            if not sp.issparse(adj):
                adj = sp.csr_matrix(adj)
            
            # Remove self-loops (standard graph cleanup)
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()

            # Construct NetworkX graph
            # Handle different versions of NetworkX
            if hasattr(nx, 'from_scipy_sparse_array'):
                G = nx.from_scipy_sparse_array(adj)
            else:
                # Fallback for older NetworkX versions
                G = nx.from_scipy_sparse_matrix(adj)
            
            # Run Louvain Algorithm
            # Using NetworkX built-in implementation to avoid 'community' package conflicts
            # Note: louvain_communities returns a list of sets of nodes
            if hasattr(nx_comm, 'louvain_communities'):
                communities = nx_comm.louvain_communities(G, seed=seed)
            else:
                # Fallback attempt if networkx is too old (requires python-louvain/community correctly installed)
                import community as community_louvain
                partition = community_louvain.best_partition(G, random_state=seed)
                # Convert partition dict to list of sets for uniform processing
                communities = [set() for _ in range(len(set(partition.values())))]
                for node, comm_id in partition.items():
                    communities[comm_id].add(node)
            
            # Convert communities (list of sets) to labels (array)
            predict_labels = np.zeros(len(true_labels), dtype=int)
            for c_id, community in enumerate(communities):
                for node in community:
                    if node < len(predict_labels): 
                        predict_labels[node] = c_id
            
            # Calculate Metrics
            # K (Number of clusters)
            k = len(communities)
            
            # Modularity
            mod = nx_comm.modularity(G, communities)
            
            # NMI and ARI (using utils.eva)
            nmi, ari = eva(true_labels, predict_labels, show_details=False)
            
            k_list.append(k)
            modularity_list.append(mod)
            nmi_list.append(nmi * 100)
            ari_list.append(ari * 100)

        except Exception as e:
            print(f"Error on {dataset} seed {seed}: {e}")

    # Convert to numpy arrays for stats
    k_list = np.array(k_list)
    modularity_list = np.array(modularity_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)

    # Report results as requested
    print(f"Results for {dataset}:")
    print(k_list.mean(), k_list.std())
    print(modularity_list.mean(), modularity_list.std())
    print(nmi_list.mean(), nmi_list.std())
    print(ari_list.mean(), ari_list.std())
    print("-" * 20)