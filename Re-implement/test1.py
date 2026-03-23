import torch
import numpy as np
import skfuzzy as fuzz
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

def eva(y_true, y_pred):
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    return nmi, ari

def clustering_step_by_step(feature, true_labels, cluster_num, device):
    print("\n--- Clustering Process Start ---")
    print(f"1. Input Feature Shape (Torch): {feature.shape}")
    
    # ---------------------------------------------------------
    # Step 1: Prepare data for scikit-fuzzy
    # ---------------------------------------------------------
    # skfuzzy expects data shape: (n_features, n_samples)
    # So we must transpose our (n_samples, n_features) input
    feature_np = feature.detach().cpu().numpy().T 
    print(f"   -> Transposed for skfuzzy:   {feature_np.shape} (Features x Samples)")

    print("\n2. Running Fuzzy C-Means (cmeans)...")
    # ---------------------------------------------------------
    # Step 2: Run Fuzzy C-Means
    # ---------------------------------------------------------
    # cntr: Cluster centers
    # u: Final fuzzy c-partitioned matrix (n_clusters, n_samples)
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data=feature_np, 
        c=cluster_num, 
        m=2.0, 
        error=0.005, 
        maxiter=1000, 
        init=None
    )
    
    print(f"   -> Centers Shape (cntr):     {cntr.shape}")
    print(f"   -> Membership Shape (u):     {u.shape} (Clusters x Samples)")
    print("   -> Raw Membership (First 2 samples):\n", u[:, :2])

    # ---------------------------------------------------------
    # Step 3: Process results back to PyTorch format
    # ---------------------------------------------------------
    print("\n3. Processing Results...")
    # Transpose u back to (n_samples, n_clusters)
    u = u.T 
    print(f"   -> Transposed Membership (u): {u.shape} (Samples x Clusters)")
    
    # Convert back to PyTorch/GPU
    centers = torch.from_numpy(cntr).float().to(device)
    predict_labels_soft = torch.from_numpy(u).float().to(device)

    # ---------------------------------------------------------
    # Step 4: Overlapping Logic (Thresholding)
    # ---------------------------------------------------------
    print("\n4. Applying Thresholding for Overlapping Labels...")
    # We normalize so the strongest membership for each node is 1.0
    # This matches the logic usually found in overlapping community detection
    u_max = predict_labels_soft.max(dim=1, keepdim=True).values
    u_norm = predict_labels_soft / (u_max + 1e-8)
    
    print("   -> Normalized Membership (First 2 samples):\n", u_norm[:2])
    
    # Apply threshold to get binary overlapping labels
    # If a node has strong membership in multiple clusters (e.g. 0.6 and 0.8), 
    # both will become 1.
    predict_labels_matrix = (u_norm > 0.5).float() 
    print("   -> Binary Labels Matrix (First 2 samples, > 0.5):\n", predict_labels_matrix[:2])

    # ---------------------------------------------------------
    # Step 5: Calculate Distances
    # ---------------------------------------------------------
    print("\n5. Calculating Squared Distances...")
    # The RL agent usually needs distance to centers for reward calculation
    feature = feature.to(device)
    dis = torch.cdist(feature, centers).pow(2)
    print(f"   -> Distance Matrix Shape:    {dis.shape}")
    print("   -> Squared Distances (First 2 samples):\n", dis[:2])

    # ---------------------------------------------------------
    # Step 6: Evaluation
    # ---------------------------------------------------------
    print("\n6. Evaluation (NMI/ARI)...")
    # For standard metrics, we flatten overlapping labels to the single "dominant" one
    dominant_labels = np.argmax(u, axis=1)
    
    nmi, ari = eva(true_labels, dominant_labels)
    print(f"   -> Scores: NMI: {nmi:.4f}, ARI: {ari:.4f}")
    
    print("--- Clustering Process End ---")
    return 100 * nmi, 100 * ari, predict_labels_matrix, centers, dis

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Create a dummy graph dataset
    # Total 10 nodes, 2 features each
    # Nodes 0-4: Cluster 1 (around [1,1])
    # Nodes 5-9: Cluster 2 (around [5,5])
    print("Generating Dummy Data...")
    torch.manual_seed(42)
    
    # Cluster 1 data
    c1 = torch.randn(5, 2) + torch.tensor([1.0, 1.0])
    # Cluster 2 data
    c2 = torch.randn(5, 2) + torch.tensor([5.0, 5.0])
    
    features = torch.cat([c1, c2], dim=0) # Shape: (10, 2)
    true_labels = np.array([0]*5 + [1]*5)

    print(f"Data created: {features.shape} samples.")

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Run visualization
    clustering_step_by_step(features, true_labels, cluster_num=2, device=device)