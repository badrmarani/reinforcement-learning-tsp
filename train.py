import torch
import os

from model import TSP
from utils import cost, draw

torch.autograd.set_detect_anomaly(True)


import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (9,9)

cp = os.path.join("checkpoints")
out = os.path.join("outputs")
if not os.path.exists(cp):
    os.makedirs(cp)
if not os.path.exists(out):
    os.makedirs(out)


dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import json
with open("args.json", "r") as f:
    args = json.load(f)

m = TSP(args["nbatchs"], args["nnodes"], args["embed_size"], args["lenPE"], args["nheads"], args["nlayersENC"], args["nlayersDEC"], mode="train").to(device)
m2 = TSP(args["nbatchs"], args["nnodes"], args["embed_size"], args["lenPE"], args["nheads"], args["nlayersENC"], args["nlayersDEC"], mode="baseline").to(device)
optim = torch.optim.Adam(m.parameters(), lr=args["lr"])


XX = torch.randn((args["nbatchs"], args["nnodes"], 2), dtype=dtype, device=device)

pm = []
pm2 = []
for epoch in range(1, args["nepochs"]+1):
    # m.train()
    l = 0
    for _ in range(args["nbatchs_per_ep"]):
        x = torch.randn((args["nbatchs"], args["nnodes"], 2), dtype=dtype, device=device)

        y, y_log_prob = m(x)
        with torch.no_grad():
            y2, _ = m2(x)
            # print(y, y2)
            y_cost, y2_cost = cost(x, y), cost(x, y2)

        loss = torch.mean((y_cost-y2_cost) * y_log_prob)

        optim.zero_grad()
        loss.backward(retain_graph=False)
        optim.step()

    # m.eval()
    y_cost_m = 0
    y2_cost_m = 0
    for _ in range(args["nbatchs_per_ev"]):
        x = torch.randn((args["nbatchs"], args["nnodes"], 2), dtype=dtype, device=device)
        with torch.no_grad():
            y, _ = m(x)
            y2, _ = m2(x)
        
        y_cost, y2_cost = cost(x, y), cost(x, y2)
        
        y_cost_m += y_cost.mean()
        y2_cost_m += y2_cost.mean()
        
    y_cost_m /= args["nbatchs_per_ep"]
    y2_cost_m /= args["nbatchs_per_ep"]

    pm.append([(epoch, y_cost_m)])
    pm2.append([(epoch, y2_cost_m)])

    if y_cost_m*1e-3 < y2_cost_m:
        m2.load_state_dict(m.state_dict())

    print(f"""
        {epoch}/{args["nepochs"]}, <eval> cost1 = {y_cost_m:.4f}, cost2 = {y2_cost_m:.4f}
    """)

    torch.save({
        "epoch": epoch,
        "loss": loss.item(),
        "plot_performance_train": pm,
        "plot_performance_baseline": pm2,
        "model_baseline": m2.state_dict(),
        "model_train": m.state_dict(),
        "optimizer": optim.state_dict(),
        }, f"{cp}/best_checkpoint.pkl"
    )

    if not epoch%1:
        with torch.no_grad():
            y, y2 = m(XX)[0], m2(XX)[0]

            fig, ax = plt.subplots(1, 2)
            draw(XX, y, ax[0], title=f"model")
            draw(XX, y2, ax[1], title=f"greedy")
            plt.savefig(f"{out}/output_{epoch}.jpg", dpi=700)
            plt.close()

plt.plot(torch.arange(args["nepochs"]), pm, label="model")
plt.plot(torch.arange(args["nepochs"]), pm2, label="greedy")
plt.legend()
plt.grid()
plt.savefig(f"{out}/output_plot_hist.jpg", dpi=700)