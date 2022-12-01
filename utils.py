import torch
import matplotlib.pyplot as plt

def cost(x, xsol):
    nbatchs, nnodes, _ = x.size()
    idkw = torch.arange(nbatchs)
    xf = x[idkw, xsol[:, 0], :]
    xprev = xf
    
    cost = torch.zeros(nbatchs)
    for i in range(1, nnodes):
        xcurr = x[idkw, xsol[:, i], :]
        cost = cost + torch.norm(xcurr-xprev, p=2)
        xprev = xcurr
    
    cost += torch.norm(xf-xprev, p=2)
    return cost

def draw(x, xsol, ax, title=None):
    ax.plot(x[0, xsol[0,:], 0], x[0, xsol[0,:], 1], "r-o")
    ax.plot(
        [x[0, xsol[0,0], 0], x[0, xsol[0,-1], 0]],
        [x[0, xsol[0,0], 1], x[0, xsol[0,-1], 1]], "r-o"
    )
    ax.set_title(title)
    ax.axis(False)