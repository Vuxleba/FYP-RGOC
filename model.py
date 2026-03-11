import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class my_model(nn.Module):
    def __init__(self, dims, act="ident"): # dims.shape = [d, 512]
        super(my_model, self).__init__()
        self.lin1 = nn.Linear(dims[0], dims[1])
        self.lin2 = nn.Linear(dims[0], dims[1])
        self.bn = nn.BatchNorm1d(dims[1])
        self.reset_parameters()

        if act == "ident":
            self.activate = lambda x: x
        if act == "sigmoid":
            self.activate = nn.Sigmoid()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        out1 = self.activate(self.lin1(x))
        out2 = self.activate(self.lin2(x))

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        return out1, out2


class my_Q_net(nn.Module):
    def __init__(self, dims):
        super(my_Q_net, self).__init__()
        self.lin1 = nn.Linear(dims[0], dims[1])
        self.lin_cluster = nn.Linear(dims[0], dims[1])
        self.lin2 = nn.Linear(dims[1], dims[2])
        self.reset_parameters()
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, cluster):
        x = self.act(F.normalize(self.lin1(x), dim=1, p=2))
        cluster = self.act(F.normalize(self.lin_cluster(cluster), dim=1, p=2))
        x = F.softmax(self.lin2(torch.cat([x, cluster], dim=0)), dim=-1)
        return x

class Dueling_Q_net(nn.Module):
    def __init__(self, dims): # dims.shape = [512, 256, 9]
        super(Dueling_Q_net, self).__init__()
        # dims: [input_dim, hidden_dim, action_dim]
        self.input_dim = dims[0]
        self.hidden_dim = dims[1]
        self.action_dim = dims[2]

        # Feature extraction (shared)
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin_cluster = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Value stream (generates scalar V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream (generates vector A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
        
        self.act = torch.nn.ReLU()

    def forward(self, x, cluster): # x.shape = [n, d], cluster.shape = [k, d]
        # Process node state
        x_emb = self.act(F.normalize(self.lin1(x), dim=1, p=2)) # x_emb.shape = [n, 256]
        # Process cluster state
        c_emb = self.act(F.normalize(self.lin_cluster(cluster), dim=1, p=2)) # c_emb.shape = [k, 256]
        
        # Combine features (using addition as they map to same hidden space)
        features = torch.cat([x_emb, c_emb], dim=0) # features.shape = [n+k, 256]
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals
    
class CAN(nn.Module):
    def __init__(self, num_nodes, num_features, num_clusters, device=torch.device('cuda')):
        super(CAN, self).__init__()
        self.device = device
        
        # Initialize F (Node Affiliations) and C (Cluster Centers)
        # F shape: [N, K] | C shape: [K, D]
        self.F_mat = nn.Parameter(torch.rand((num_nodes, num_clusters), device=device))
        self.C_mat = nn.Parameter(torch.rand((num_clusters, num_features), device=device))

    def forward(self):
        # Reconstruct X using affiliations and centers: X ~ Relu(F) @ Relu(C)
        X_hat = torch.matmul(torch.relu(self.F_mat), torch.relu(self.C_mat))
        return X_hat

    def fit(self, X, max_iter=100, lr=0.05):
        # Shift latent space to be non-negative if necessary
        X_min = X.min()
        if X_min < 0:
            X_pos = X - X_min
        else:
            X_pos = X
        
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Optimization loop
        for i in range(max_iter):
            optimizer.zero_grad()
            
            X_hat = self.forward()
            
            # Loss: Reconstruction Error + Sparsity Constraint on Affiliations
            loss = F.mse_loss(X_hat, X_pos) + 0.01 * torch.relu(self.F_mat).mean()
            
            loss.backward()
            optimizer.step()
        
        # Return shift value for post-processing
        return X_min
    
    def predict(self, X, X_min, threshold=0.5):
        # Get final affiliations
        F_final = torch.relu(self.F_mat).detach()
        C_final = torch.relu(self.C_mat).detach()

        # Normalize so strongest affiliation is 1.0
        F_max = F_final.max(dim=1, keepdim=True).values + 1e-8
        F_norm = F_final / F_max

        # Binary labeling
        binary_labels = (F_norm > threshold).float()

        # Restore centers to original scale for distance calculation
        if X_min < 0:
            C_final = C_final + X_min
        
        # Calculate distances (for RL reward)
        dis = torch.cdist(X, C_final, p=2).pow(2)

        return binary_labels.cpu(), C_final, dis