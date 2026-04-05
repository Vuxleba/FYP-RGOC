import os
import torch
import numpy as np
import scipy.sparse as sp
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator

# Assuming utils.py is in the same directory
from utils import load_graph_data, setup_seed, preprocess_graph

warnings.filterwarnings("ignore")

# Hyperparameters mapped from your source code
HYPERPARAMS = {
    "facebook_107":  {"n_input": -1,  "threshold": 0.95, "gnnlayers": 2},
    "facebook_348":  {"n_input": 128, "threshold": 0.8,  "gnnlayers": 5},
    "facebook_414":  {"n_input": -1,  "threshold": 0.8,  "gnnlayers": 3},
    "facebook_686":  {"n_input": -1,  "threshold": 0.6,  "gnnlayers": 4},
    "facebook_698":  {"n_input": -1,  "threshold": 0.8,  "gnnlayers": 2},
}

def run_elbow_method(dataset_name, X, max_k=30):
    """
    Runs the Elbow Method for K-Means clustering, detects the elbow programmatically,
    and saves the plot.
    """
    inertias = []
    K_range = range(2, max_k + 1)
    
    # 1. Calculate WCSS (Inertia) for each K
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
    # 2. Programmatically find the Elbow using KneeLocator
    kn = KneeLocator(list(K_range), inertias, curve='convex', direction='decreasing')
    optimal_k = kn.knee
    
    if optimal_k is None:
        print(f"  ⚠️ Could not find a clear elbow for {dataset_name}!")
        optimal_k = "Ambiguous"
    else:
        print(f"  ✅ Elbow Detected at K = {optimal_k}")
        
    # 3. Plot the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, 'bo-', markersize=6, linewidth=2)
    
    if isinstance(optimal_k, int):
        plt.axvline(x=optimal_k, color='r', linestyle='--', 
                    label=f'Algorithm Detected Elbow (K={optimal_k})')
        
    plt.title(f'Elbow Method for {dataset_name} (Smoothed Features)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.xticks(list(K_range)[::2]) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.savefig(f'elbow_plot_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close() # Close to prevent overlapping plots
    
    return optimal_k


def run_elbow_on_facebook_datasets():
    datasets = list(HYPERPARAMS.keys())
    print(f"Running Elbow Method on {len(datasets)} Facebook datasets...")

    summary_results = []

    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")
        
        # Fetch dataset-specific params
        n_input = HYPERPARAMS[dataset_name]["n_input"]
        gnnlayers = HYPERPARAMS[dataset_name]["gnnlayers"]
        
        try:
            # Load Data
            X, y, A = load_graph_data(dataset_name, show_details=False)
            features = X
            adj = sp.csr_matrix(A)
            
            # Dimensionality Reduction (Dynamic PCA)
            if n_input != -1:
                pca = PCA(n_components=n_input, random_state=42)
                features = pca.fit_transform(features)
                
            # Adjacency Preprocessing
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()
            adj_norm_s = preprocess_graph(adj, gnnlayers, norm='sym', renorm=True)
            
            # Laplacian Smoothing
            sm_fea_s = sp.csr_matrix(features).toarray()
            for a in adj_norm_s:
                sm_fea_s = a.dot(sm_fea_s)

            # Extract True K (length of the ground truth labels list)
            true_k = len(y) 
            
            # Run Elbow Method on the smoothed features
            predicted_k = run_elbow_method(dataset_name, sm_fea_s, max_k=30)
            
            summary_results.append({
                'dataset': dataset_name,
                'true_k': true_k,
                'predicted_k': predicted_k
            })
            
        except Exception as e:
            print(f"  ❌ Error processing {dataset_name}: {e}")

    # Print Final Summary Table
    print("\n==========================================")
    print("=== Final Elbow Method Predictions ===")
    print("==========================================")
    print(f"{'Dataset':<15} | {'True K':<8} | {'Elbow Predicted K'}")
    print("-" * 45)
    for res in summary_results:
        print(f"{res['dataset']:<15} | {res['true_k']:<8} | {res['predicted_k']}")
    print("==========================================")


if __name__ == "__main__":
    setup_seed(42) # Ensure reproducibility for PCA and K-Means
    run_elbow_on_facebook_datasets()