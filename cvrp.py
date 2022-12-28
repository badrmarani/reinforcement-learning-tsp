import torch
from torch import nn

from tsp import MultiHeadAttention

dtype = torch.float
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_size,
        num_heads,
    ):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.attention = nn.ModuleList(
            [MultiHeadAttention(num_heads, embed_size) for _ in range(num_layers)]
        )
        self.bn = nn.ModuleList([nn.BatchNorm1d(embed_size) for _ in range(num_layers)])

        self.embed_node0 = nn.Linear(2, embed_size)
        self.embed_nodei = nn.Linear(3, embed_size)


    def forward(self, node_loc, demand):
        # node_loc -> size(num_batchs, num_nodes+1, 3)
        # demand -> size(num_batchs, num_nodes)
        node0 = node_loc[:, 0, :2].unsqueeze(1)  # size(num_batchs, 1, 2)
        nodei = node_loc[:, 1:, :]  # size(num_batchs, num_nodes, 2)

        if len(demand.size()) < 3 or demand.size(-1) != 1:
            demand = demand.unsqueeze(-1)
        nodei = torch.cat((nodei, demand), dim=-1)  # size(num_batchs, num_nodes, 3)

        h0 = self.embed_node0(node0) # size(num_batchs, 1, embed_size)
        hi = self.embed_nodei(nodei) # size(num_batchs, num_nodes, embed_size)
        h = torch.cat((h0,hi), dim=1) # size(num_batchs, num_nodes + 1, embed_size)
        size = h.size()

        for i in range(self.num_layers):
            h = h + self.attention[i](h) # size(num_batchs * num_nodes, embed_size)
            h = h.view(-1, size[-1])
            h = self.bn[i](h)
            h = h.view(*size)  # size(num_batchs, num_nodes, embed_size)

        return h  # size(num_batchs, num_nodes, embed_size)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
    

    def forward(self, x, P, clip):
        pass



num_nodes = 4
batch_size = 2
capacity = 30

x = torch.rand(size=(batch_size, num_nodes + 1, 2), dtype=dtype)
di = torch.randint(1, 10, size=(batch_size, num_nodes, 1), dtype=dtype)

enc = Encoder(5, 20, 5)
print(enc(x, di).size())
