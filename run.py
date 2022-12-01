import torch

from model import TSP
from utils import draw

import matplotlib.pyplot as plt


dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = r"checkpoints/best_checkpoint.pkl"
checkpoint = torch.load(filename, map_location=device)

import json
with open("args.json", "r") as f:
    args = json.load(f)


hist_m = checkpoint["plot_performance_train"]
hist_m2 = checkpoint["plot_performance_baseline"]

m = TSP(args["nbatchs"], args["nnodes"], args["embed_size"], args["lenPE"], args["nheads"], args["nlayersENC"], args["nlayersDEC"], mode="train").to(device)
m2 = TSP(args["nbatchs"], args["nnodes"], args["embed_size"], args["lenPE"], args["nheads"], args["nlayersENC"], args["nlayersDEC"], mode="baseline").to(device)


m.load_state_dict(checkpoint["model_train"])
m2.load_state_dict(checkpoint["model_baseline"])

x = torch.randn((args["nbatchs"], args["nnodes"], 2), dtype=dtype, device=device)


mhist = checkpoint["plot_performance_train"]
m2hist = checkpoint["plot_performance_baseline"]


# epochs = [x[0][0] for x in mhist]
# mh = [x[0][1] for x in mhist]
# m2h = [x[0][1] for x in m2hist]
# plt.plot(epochs, mh, label="train")
# plt.plot(epochs, m2h, label="baseline")
# plt.legend()
# plt.grid()
# plt.show()

draw(x, m(x)[0], plt)
plt.show()