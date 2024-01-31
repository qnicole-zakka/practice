# gnn in pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNN, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        x = F.relu(self.linear1(x))
        x = torch.matmul(adj, x)
        x = self.linear2(x)
        return x


# load data with pytorch dataset
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html



if __name__ == '__main__':
    x = torch.randn(10, 8)
    adj = torch.randn(10, 10)
    model = GNN(8, 16, 7)
    out = model(x, adj)
    print(out)
 

od = OrderedDict()

for (k, v) in od.items():
    print(k, v)