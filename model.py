import torch
import torch.nn as nn
import torch.nn.functional as F


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