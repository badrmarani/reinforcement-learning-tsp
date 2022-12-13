from model import TSPNet
from utils import draw, Trainer
import torch

import matplotlib.pyplot as plt
import yaml

with open("args.yml", "r", encoding="utf-8") as f:
    args = yaml.safe_load(f)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TSPNet(args["embed_size"], args["num_heads"], greedy=False).to(device)
greedy = TSPNet(args["embed_size"], args["num_heads"], greedy=True).to(device)
greedy.load_state_dict(model.state_dict())
optim = torch.optim.Adam(model.parameters(), lr=args["lr"])

history = {"train": [], "test": []}
T = Trainer(args, dtype, device)
for epoch in range(args["num_epochs"]):
    train_loss = T.train(epoch, (model, greedy), optim)
    test_loss = T.test(epoch, (model, greedy), optim)
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
            xx = torch.rand(size=(1, args["num_nodes"], 2), device=device, dtype=dtype)
            sol1, sol2 = model(xx)[0], greedy(xx)[0]
            draw(xx, sol1, plt)
            plt.title("model")
            plt.axis(False)
            plt.savefig(f"results/sample_viz_{epoch}.jpg", dpi=700)
            plt.close()