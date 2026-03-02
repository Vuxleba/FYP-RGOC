import torch
import torch.nn as nn
import torch.nn.functional as F

class my_Model(nn.Module):
    def __init__(self, dims, act="ident"):
        super(my_Model, self).__init__()
        self.lin1 = nn.Linear(dims[0], dims[1])
        self.lin2 = nn.Linear(dims[0], dims[1])
        self.bn = nn.BatchNorm1d(dims[1])
        self.reset_parameters()

        if act == "sigmoid":
            self.activation = nn.Sigmoid()
        if act == "ident":
            self.activation = lambda x : x
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def forward(self):
        out1 = self.activation(self.lin1)
        out2 = self.activation(self.lin2)

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        return out1, out2

class my_Q_net(nn.Module):
    def __init__(self, dims):
        super(my_Q_net, self).__init__()
        self.lin_z = nn.Linear(dims[0], dims[1])
        self.lin_c = nn.Linear(dims[0], dims[1])
        self.lin_o = nn.Linear(dims[1], dims[2])
        self.reset_parameters()
        self.act = nn.ReLU()

    def reset_parameters(self):
        self.lin_z.reset_parameters()
        self.lin_c.reset_parameters()

    def forward(self, x, c):
        x = self.act(F.normalize(self.lin_z(x), dim=1, p=2))
        c = self.act(F.normalize(self.lin_c(c), dim=1, p=2))
        x = F.softmax(self.lin_o(torch.concat([x,c], dim=0)), dim=-1)
        return x
         