import torch
import os

from model import TSP
from utils import cost, draw

torch.autograd.set_detect_anomaly(True)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    "nepochs": 20,
    "nbatchs": 2,
    "nbatchs_per_ep": 5,
    "nnodes": 7,
    "nheads": 4,
    "embed_size": 4,
    "nlayersENC": 2,
    "nlayersDEC": 2,
    "lenPE": 1000,
    "lr": 2e-3,
}



m = TSP(args["nbatchs"], args["nnodes"], args["embed_size"], args["lenPE"], args["nheads"], args["nlayersENC"], args["nlayersDEC"]).to(device)
m2 = TSP(args["nbatchs"], args["nnodes"], args["embed_size"], args["lenPE"], args["nheads"], args["nlayersENC"], args["nlayersDEC"]).to(device)
optim = torch.optim.Adam(m.parameters(), lr=args["lr"])

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
            y_cost, y2_cost = cost(x, y), cost(x, y2)

        loss = torch.mean((y_cost-y2_cost) * y_log_prob)

        optim.zero_grad()
        loss.backward(retain_graph=False)
        optim.step()

        # l+=loss.item()

    # m.eval()
    y_cost_m = 0
    y2_cost_m = 0
    for _ in range(args["nbatchs_per_ep"]):
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

    if y_cost_m < y2_cost_m:
        m2.load_state_dict(m.state_dict())

    print(f"""
        {epoch}/{args["nepochs"]}, <eval> cost1 = {y_cost_m:.4f}, cost2 = {y2_cost_m:.4f}
    """)

    cp = os.path.join("checkpoints")
    if not os.path.exists(cp):
        os.makedirs(cp)
    
    if not epoch%5:
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
