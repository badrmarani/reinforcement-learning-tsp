import torch
from torch import nn
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class sinkhorn(nn.Module):
    def __init__(self,
        niter:int,
        reg:float,

    ) -> None:
        super().__init__()

        self.niter = niter
        self.reg = reg

    def forward(self, a, b):
        pass

class encoder(nn.Module):
    def __init__(self,
        embed_size:int = 128,
        nheads:int = 8,
        in_size: int = 2,
        ninstances: int = 10,
    ) -> None:
        super(encoder, self).__init__()

        self.IN_EMBED = nn.Linear(in_size, embed_size)
        
        # embed_size must be divisible by nheads
        self.MHA = nn.MultiheadAttention(
                embed_size, nheads, batch_first=True, kdim=embed_size, vdim=embed_size)
        self.FF = nn.Linear(embed_size, embed_size)
        self.BN = nn.BatchNorm1d(ninstances)

    def forward(self, x):
        h = self.IN_EMBED(x).squeeze(0)
1

        N = 1
        for _ in range(N):
            # mha + residual_connection + batch_norm
            attn, _ = self.MHA(h, h, h)
            h += attn
            h = self.BN(h)

            # ff + residual_connection + batch_norm
            h += self.FF(h)
            h = self.BN(h)

        hbar = h.mean(dim=2)
        print(h.size(), hbar.size())
        return h, hbar


x = torch.randn((3, 10, 2), device=device, dtype=dtype)
att = encoder(ninstances=10)


# print("before", x[0, 0, :], x[0, 0, :].size())
out = att(x)
# print("after", out[0, 0, :], out, out.size())



# print(x, x.size())
# x = x.transpose(1,2)
# print(x, x.size())

# print(x[0, 0, :])
