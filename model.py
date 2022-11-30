import torch
from torch import nn, distributions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(
        self,
        nlayers,
        embed_size,
        nheads,
    ) -> None:
        super(Encoder, self).__init__()
        self.nlayers = nlayers

        self.mha = nn.ModuleList(
            nn.MultiheadAttention(embed_size, nheads, bias=False, batch_first=True)
            for _ in range(nlayers)
        )
        self.lin = nn.ModuleList(
            nn.Linear(embed_size, embed_size) for _ in range(nlayers)
        )
        self.lin1 = nn.ModuleList(
            nn.Linear(embed_size, embed_size) for _ in range(nlayers)
        )
        self.lin2 = nn.ModuleList(
            nn.Linear(embed_size, embed_size) for _ in range(nlayers)
        )
        self.bn = nn.ModuleList(nn.BatchNorm1d(embed_size) for _ in range(nlayers))

    def forward(self, h):
        for i in range(self.nlayers):
            h = h + self.mha[i](h, h, h)[0]
            h = h.permute(1, 2, 0).contiguous()
            h = self.bn[i](h).transpose(1, 2).contiguous()

            h = h + self.lin2[i](torch.relu(self.lin1[i](h)))
            h = h.permute(1, 2, 0).contiguous()
            h = self.bn[i](h).transpose(1, 2).contiguous()

        return h


class MHA(nn.Module):
    def __init__(
        self,
        nheads,
    ) -> None:
        super(MHA, self).__init__()
        self.nheads = nheads

    def forward(self, query, key, value, mask=None, clip=None):
        nbatchs, nnodes, embed_size = query.size()
        assert (
            not embed_size % self.nheads
        ), "The embedding size has to be a multiple of the number of heads."

        query, key, value = [
            x.transpose(1, 2)
            .contiguous()
            .view(nbatchs * self.nheads, embed_size // self.nheads, nh)
            .transpose(1, 2)
            .contiguous()
            for x, nh in zip(
                (query, key, value), (query.size(1), key.size(1), value.size(1))
            )
        ]

        ktmp = key.transpose(1, 2).contiguous()
        qk = torch.bmm(query, ktmp)
        qk = qk / (embed_size // self.nheads) ** 0.5
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.repeat_interleave(repeats=self.nheads, dim=0)
            qk = qk.masked_fill(mask == 1, float("-1e9"))

        if clip is not None:
            qk = clip * torch.tanh(qk)

        attn_prob = torch.softmax(qk, dim=-1)

        attn = torch.bmm(attn_prob, value)

        attn = attn.view(nbatchs, nnodes, embed_size)
        attn_prob = attn_prob.view(nbatchs, self.nheads, 1, key.size(1)).mean(dim=1)
        return attn, attn_prob


class ARD(nn.Module):
    def __init__(
        self,
        embed_size,
        nheads,
    ) -> None:
        super(ARD, self).__init__()
        self.kprev = None
        self.vprev = None
        self.lin = nn.ModuleList(nn.Linear(embed_size, embed_size) for _ in range(8))
        self.mha = MHA(nheads).to(device)
        self.bn = nn.ModuleList(nn.LayerNorm(embed_size) for _ in range(3))

    def reset(self):
        self.kprev = None
        self.vprev = None

    def forward(self, ht, key, value, mask):
        ht = ht.view(ht.size(0), 1, -1)
        q = self.lin[0](ht)
        k = self.lin[1](ht)
        v = self.lin[2](ht)

        if self.kprev is None:
            self.kprev = k
            self.vprev = v
        else:
            self.kprev = torch.cat((self.kprev, k), dim=1)
            self.vprev = torch.cat((self.vprev, v), dim=1)

        # STEP (2)
        ht = ht + self.lin[3](self.mha(q, self.kprev, self.vprev, mask=None)[0])
        ht = self.bn[0](ht.squeeze())
        ht = ht.view(ht.size(0), 1, -1)

        # STEP (3)
        query = self.lin[4](ht)
        ht = ht + self.lin[5](self.mha(query, key, value, mask, clip=None)[0])
        ht = self.bn[1](ht.squeeze(1))
        ht = ht.view(ht.size(0), 1, -1)

        ht = ht + self.lin[6](torch.relu(self.lin[7](ht)))
        ht = self.bn[2](ht.squeeze(1))
        return ht


class Decoder(nn.Module):
    def __init__(
        self,
        nlayers,
        embed_size,
        nheads,
    ) -> None:
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.nlayers = nlayers

        self.lin = nn.Linear(embed_size, embed_size)
        self.ard = nn.ModuleList(ARD(embed_size, nheads).to(device) for _ in range(nlayers))
        self.mha = MHA(nheads).to(device)

    def reset(self):
        for i in range(self.nlayers):
            self.ard[i].reset()

    def forward(self, query, key, value, mask, clip):
        for i in range(self.nlayers):
            kl = key[:, :, (i) * self.embed_size : (i + 1) * self.embed_size]
            vl = value[:, :, (i) * self.embed_size : (i + 1) * self.embed_size]

            if i == self.nlayers - 1:
                ht = self.lin(ht).unsqueeze(1)
                _, ht_prob = self.mha(ht, kl, vl, mask, clip=clip)
            else:
                ht = self.ard[i](query, kl, vl, mask)
        return ht_prob.squeeze(1)


class TSP(nn.Module):
    def __init__(
        self,
        nbatchs,
        nnodes,
        embed_size,
        lenPE,
        nheads,
        nlayersENC,
        nlayersDEC,
    ) -> None:
        super(TSP, self).__init__()
        self.embed_size = embed_size
        self.nnodes = nnodes
        self.nbatchs = nbatchs
        self.lenPE = lenPE

        self.enc = Encoder(nlayersENC, embed_size, nheads).to(device)
        self.dec = Decoder(nlayersDEC, embed_size, nheads).to(device)

        self.Wk = nn.Linear(embed_size, embed_size * nlayersDEC)
        self.Wv = nn.Linear(embed_size, embed_size * nlayersDEC)

        self.lin = nn.Linear(2, embed_size)

    def _positional_encoding(self):
        t = torch.arange(0, self.lenPE).unsqueeze(1)
        fi = torch.exp(
            torch.arange(0, self.embed_size, 2)
            * (-torch.log(torch.tensor(10_000)) / self.embed_size)
        )

        pe = torch.zeros(self.lenPE, self.embed_size)
        pe[:, 0::2] = torch.sin(t * fi)
        pe[:, 1::2] = torch.cos(t * fi)
        return pe

    def forward(self, x):
        idkw = torch.arange(self.nbatchs)

        henc = self.lin(x)
        henc = self.enc(henc)

        sol, sol_prob = [], []
        pe = self._positional_encoding()

        str_idx = torch.tensor(self.nnodes - 1).expand(self.nbatchs)

        # INTIALIZATION OF STEP (1)
        ht = pe[0].repeat(self.nbatchs, 1) + henc[idkw, str_idx, :]


        k = self.Wk(henc)
        v = self.Wv(henc)
        mask = torch.zeros((self.nbatchs, self.nnodes))
        mask[idkw, str_idx] = True


        self.dec.reset()

        for t in range(self.nnodes):
            ht = self.dec(
                ht, k, v, mask, clip=10
            )  # the probabilities of each node being the next one

            nxnode = distributions.Categorical(ht).sample()
            sol.append(nxnode)
            sol_prob.append(torch.log(ht[idkw, nxnode]))


            ht = henc[idkw, nxnode, :]
            ht = ht + pe[t + 1].expand(self.nbatchs, self.embed_size)

            mask[idkw, nxnode] = True
        

        sol_prob = torch.stack(sol_prob, dim=1).sum(dim=1)
        sol = torch.stack(sol, dim=1)
        return sol, sol_prob
