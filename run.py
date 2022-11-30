import torch

from model import TSP
from utils import draw

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = r"checkpoints/best_checkpoint.pkl"
checkpoint = torch.load(filename, map_location=device)

args = {
    "nepochs": 20,
    "nbatchs": 1,
    "nbatchs_per_ep": 5,
    "nnodes": 7,
    "nheads": 4,
    "embed_size": 4,
    "nlayersENC": 2,
    "nlayersDEC": 2,
    "lenPE": 1000,
    "lr": 2e-3,
}

hist_m = checkpoint["plot_performance_train"]
hist_m2 = checkpoint["plot_performance_baseline"]

m = TSP(args["nbatchs"], args["nnodes"], args["embed_size"], args["lenPE"], args["nheads"], args["nlayersENC"], args["nlayersDEC"]).to(device)
m2 = TSP(args["nbatchs"], args["nnodes"], args["embed_size"], args["lenPE"], args["nheads"], args["nlayersENC"], args["nlayersDEC"]).to(device)


# m.load_state_dict(checkpoint["model_train"])
# m2.load_state_dict(checkpoint["model_baseline"])

x = torch.randn((args["nbatchs"], args["nnodes"], 2), dtype=dtype, device=device)
print(m(x)[0])

