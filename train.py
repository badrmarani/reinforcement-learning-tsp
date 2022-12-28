from tsp import TSPNet
from utils import draw, Trainer
import torch

import matplotlib.pyplot as plt
import yaml

import os
import shutil

if os.path.exists("results/"):
    shutil.rmtree("results/")
    os.makedirs("results/")
    os.makedirs("results/checkpoints")

with open("args.yml", "r", encoding="utf-8") as f:
    args = yaml.safe_load(f)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TSPNet(
    args["embed_size"], args["num_heads"], reg=2, niters=10, greedy=False
).to(device)
greedy = TSPNet(
    args["embed_size"], args["num_heads"], reg=2, niters=10, greedy=True
).to(device)
greedy.load_state_dict(model.state_dict())
optim = torch.optim.Adam(model.parameters(), lr=args["lr"])


xx = torch.rand(size=(1, args["num_nodes"], 2), device=device, dtype=dtype)

history = {"train": [], "test": []}
T = Trainer(args, dtype, device)
for epoch in range(1, args["num_epochs"] + 1):
    model, greedy, train_loss = T.train(
        epoch, args["num_epochs"], (model, greedy), optim
    )
    model, greedy, test_loss = T.test(
        epoch, args["num_epochs"], args["batch_size_val"], (model, greedy), optim
    )
    history["test"].append(test_loss)
    history["train"].append(train_loss)

    if not epoch % args["save_interval"]:
        torch.save(
            {
                "loss_history": history,
                "args": args,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
            },
            f="results/checkpoints/checkpoint_{}.pkl".format(epoch),
        )

        with torch.no_grad():
            sol1, sol2 = model(xx)[0], greedy(xx)[0]
            draw(xx, sol1, plt)
            plt.title("{}/{}".format(epoch, args["num_epochs"]))
            plt.axis(False)
            plt.savefig(f"results/sample_viz_{epoch}.jpg", dpi=700)
            plt.close()

plt.figure(figsize=(20, 20))
plt.plot(history["train"], label="training loss")
plt.plot(history["test"], label="test loss")
plt.grid(True)
plt.legend()
plt.savefig("results/loss.jpg", dpi=700)
plt.close()
