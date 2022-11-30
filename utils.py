import torch
import matplotlib.pyplot as plt

def cost(x, xsol):
    nbatchs, nnodes = xsol.size()
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

def draw(x, xsol):
    plt.figure(figsize=(3,3))
    plt.plot(solution[:,0], solution[:,1], "r-o")
    plt.plot(
        [solution[0,0], solution[-1,0]],
        [solution[0,1], solution[-1,1]], "r-o")
    plt.title(f"Total distance {cost(solution):.3f}")
    plt.axis(False)
    plt.show()
