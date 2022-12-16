import numpy as np
import torch
from torch import nn

import sys

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(seed=1234)
torch.cuda.manual_seed(seed=1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np

np.random.seed(1234)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        q_embed_size,
        k_embed_size=None,
        v_embed_size=None,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.q_embed_size = q_embed_size
        self.k_embed_size = k_embed_size if k_embed_size is not None else q_embed_size
        self.v_embed_size = v_embed_size if v_embed_size is not None else q_embed_size
        self.prjQ = nn.Linear(self.q_embed_size, 16 * num_heads, bias=False)
        self.prjK = nn.Linear(self.k_embed_size, 16 * num_heads, bias=False)
        self.prjVinp = nn.Linear(self.v_embed_size, 16 * num_heads, bias=False)
        self.prjVout = nn.Linear(16 * num_heads, self.v_embed_size, bias=False)

    def forward(self, Q, K=None, V=None, mask=None, clip=None):
        if K is None and V is None:
            K = Q.detach().clone()
            V = Q.detach().clone()

        num_batchs, num_nodes, _ = Q.size()
        # assert (
        # 	not num_batchs%self.num_heads
        # ), "The embedding size has to be a multiple of the number of heads."

        # size(num_batchs, num_heads, num_nodes, 16)
        Q = torch.stack(torch.chunk(self.prjQ(Q), self.num_heads, dim=-1), dim=1)
        K = torch.stack(torch.chunk(self.prjK(K), self.num_heads, dim=-1), dim=1)
        V = torch.stack(torch.chunk(self.prjVinp(V), self.num_heads, dim=-1), dim=1)

        U = torch.matmul(Q, K.transpose(2, 3)) / 16**0.5

        if clip is not None:
            U = clip * torch.tanh(U)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            U = U.masked_fill(mask, float("-inf"))

        U_prob = torch.softmax(U, dim=-1)
        U = torch.matmul(U_prob, V)
        U = U.transpose(1, 2).reshape(num_batchs, num_nodes, self.num_heads * 16)
        U = self.prjVout(U)
        return U


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_size,
        num_heads,
    ) -> None:
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Linear(2, embed_size)
        self.attention = nn.ModuleList(
            [MultiHeadAttention(num_heads, embed_size) for _ in range(num_layers)]
        )
        self.bn = nn.ModuleList([nn.BatchNorm1d(embed_size) for _ in range(num_layers)])

        self.lina = nn.Linear(embed_size, embed_size, bias=False)
        self.linb = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x, clip=10):
        h = self.embedding(x)  # size(num_batchs, num_nodes, embed_size)
        size = h.size()
        for i in range(self.num_layers):
            h = h + self.attention[i](h)
            # size(num_batchs * num_nodes, embed_size)
            h = h.view(-1, size[-1])
            h = self.bn[i](h)
            h = h.view(*size)  # size(num_batchs, num_nodes, embed_size)

        return h  # size(num_batchs, num_nodes, embed_size)


class Decoder(nn.Module):
    def __init__(
        self,
        embed_size,
        num_heads,
        greedy,
    ) -> None:
        super(Decoder, self).__init__()
        self.greedy = greedy
        # self.initK = nn.Parameter(
        #     torch.empty(size=(1, 1, embed_size), device=device, dtype=dtype).uniform_()
        # )
        # self.initV = nn.Parameter(
        #     torch.empty(size=(1, 1, embed_size), device=device, dtype=dtype).uniform_()
        # )

        self.lin = nn.Linear(2, embed_size, bias=False)

        self.prjK = nn.Linear(embed_size, embed_size, bias=False)
        self.attention = MultiHeadAttention(
            num_heads, embed_size * 3, embed_size, embed_size
        )

    def _apply_softmax(self, x):
        return torch.nn.functional.softmax(x, dim=-1)

    def forward(self, x, P, clip):
        num_batchs, num_nodes, embed_size = x.size()
        indices = (x.sum(dim=-1, keepdim=True).argsort(dim=1, descending=True)).squeeze(
            -1
        )[:, 0]

        u = P[torch.arange(P.size(0)), indices].unsqueeze(1)

        mask = torch.zeros(
            size=(num_batchs, num_nodes), device=device, dtype=dtype
        ).bool()
        logprob = 0.0
        visited_nodes = []
        for i in range(x.size(1)):
            u = u.masked_fill(mask.unsqueeze(1), float("-inf"))

            if self.greedy:
                next_node = u.argmax(dim=-1)
            else:
                m = torch.distributions.Categorical(logits=u)
                next_node = m.sample()
                logprob += m.log_prob(next_node)

            visited_nodes.append(next_node)
            mask = mask.scatter(1, next_node, True)

            next_node = next_node.unsqueeze(-1).repeat(1, 1, embed_size)
            u = P[torch.arange(P.size(0)), next_node[..., 0].squeeze(-1)].unsqueeze(1)

        visited_nodes = torch.cat(visited_nodes, -1)
        return visited_nodes, logprob


class Sinkhorn(nn.Module):
    def __init__(self, reg: float, niters: int) -> None:
        super(Sinkhorn, self).__init__()
        self.reg = reg
        self.niters = niters

    def forward(self, cost):
        num_batchs, num_nodes, _ = cost.size()
        K = torch.exp(-self.reg * cost)
        u = torch.ones(size=(num_batchs, num_nodes)) / num_nodes
        v = torch.ones(size=(num_batchs, num_nodes)) / num_nodes
        for _ in range(self.niters):
            v = 1 / torch.einsum("ki,kij->kj", u, K)
            u = 1 / torch.einsum("kij,kj->ki", K, v)
        P = torch.einsum("ki,kij,kj->kij", u, K, v)
        P = torch.log(P)
        return P


class TSPNet(nn.Module):
    def __init__(
        self,
        embed_size,
        num_heads,
        reg,
        niters,
        greedy,
    ) -> None:
        super(TSPNet, self).__init__()
        self.encoder = Encoder(10, embed_size, num_heads)
        self.decoder = Decoder(embed_size, num_heads, greedy=greedy)

        self.lina = nn.Linear(embed_size, embed_size, bias=False)
        self.linb = nn.Linear(embed_size, embed_size, bias=False)

        self.sinkhorn = Sinkhorn(reg, niters)

    def _make_heatmap(self, x, clip=10):
        # x <- size(num_batchs, num_nodes, embed_size)
        A = self.lina(x)
        B = self.linb(x)
        M = torch.bmm(A, B.transpose(-1, -2)) / A.size(-1)
        Ptanh = torch.tanh(M) * clip
        return Ptanh  # size(num_batchs, num_nodes, num_nodes)

    def forward(self, instances):
        H = self.encoder(instances)
        P = self._make_heatmap(H)
        P = self.sinkhorn(P)

        solutions, log_probs = self.decoder(H, P, clip=10)
        return solutions, log_probs
