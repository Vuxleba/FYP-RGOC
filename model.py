import torch
import torch.nn as nn
import torch.nn.functional as F

class my_model(nn.Module):
    """
    Core graph neural network framework containing twin linear pathways.
    Used to process graph inputs into encoded embeddings.
    """
    def __init__(self, dims, act="relu"): # dims.shape = [d, 512]
        """
        Initialize the twin pathway model.
        Args:
            dims: List mapping input -> output dimension sizes.
            act: The activation function type ('ident', 'sigmoid', 'relu').
        """
        super(my_model, self).__init__()
        self.lin1 = nn.Linear(dims[0], dims[1])
        self.lin2 = nn.Linear(dims[0], dims[1])
        self.reset_parameters()

        if act == "ident":
            self.activate = lambda x: x
        if act == "sigmoid":
            self.activate = nn.Sigmoid()
        if act == "relu":
            self.activate = nn.ReLU()
    def reset_parameters(self):
        """Reset the parameters of linear layers."""
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        """
        Forward logic processing nodes through both linear pathways.
        Outputs normalized embeddings.
        """
        out1 = self.activate(self.lin1(x))
        out2 = self.activate(self.lin2(x))

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        return out1, out2


class my_Q_net(nn.Module):
    """
    Q-Network for Reinforcement Learning to evaluate clustering assignments.
    Takes cluster pools and raw nodes, combining to output Q-values.
    """
    def __init__(self, dims):
        """
        Setup linear layer representations.
        Args:
            dims: Dimension scale containing [input_dim, hidden_dim, out_dim].
        """
        super(my_Q_net, self).__init__()
        self.lin_z = nn.Linear(dims[0], dims[1])
        self.lin_cluster = nn.Linear(dims[0], dims[1])
        self.lin_out = nn.Linear(dims[1], dims[2])
        self.reset_parameters()
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        """Reset the parameters of inner linear paths."""
        self.lin_z.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, x, cluster):
        """
        Forward process taking aggregated node data and clusters
        to generate a combined action Q-value.
        """
        # 1. Global Pooling: Aggregate node embeddings [N, D] -> [1, D]
        x_pooled = x.mean(dim=0, keepdim=True)
        # Aggregate cluster centroids [K, D] -> [1, D]
        cluster_pooled = cluster.mean(dim=0, keepdim=True)

        # 2. Process features
        x_emb = self.act(F.normalize(self.lin_z(x_pooled), dim=1, p=2)) # [1, 256]
        cluster_emb = self.act(F.normalize(self.lin_cluster(cluster_pooled), dim=1, p=2)) # [1, 256]
        
        # 3. Combine representations (summing to keep dimension [1, 256])
        combined = x_emb + cluster_emb
        
        # 4. Output Q-values for actions
        out = self.lin_out(combined)
        
        # Return a 1D vector of Q-values 
        return out.squeeze()