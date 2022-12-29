import os
import shutil
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats
from torch import nn

from model import TSP
from utils import cost

if os.path.exists("results/"):
    shutil.rmtree("results/")
else:
    os.makedirs("results/")
os.makedirs("results/checkpoints")


dtype = torch.float


class TSPSolver(nn.Module):
    def __init__(self, args):
        super(TSPSolver, self).__init__()
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.batch_size_val = args.batch_size_val
        self.num_instances_per_batch = args.num_instances_per_batch
        self.num_nodes = args.num_nodes
        self.num_heads = args.num_heads
        self.embed_size = args.embed_size
        self.lr = args.lr
        self.save_interval = args.save_interval
        self.reg = args.reg
        self.num_iters = args.num_iters
        self.dtype = torch.float
        self.device = torch.device(args.device)

        self.loss_history_train = []
        self.loss_history_test = []

        self.solver = TSP(
            self.embed_size,
            self.num_heads,
            self.reg,
            self.num_iters,
            greedy=False,
        )

        self.greedy = TSP(
            self.embed_size,
            self.num_heads,
            self.reg,
            self.num_iters,
            greedy=True,
        )

        if self.device == "cuda" and args.parallel:
            self.solver = nn.DataParallel(self.solver)
        self.solver.to(self.device)
        self.greedy.to(self.device)

        self.greedy.load_state_dict(self.solver.state_dict())
        self.optimizer = torch.optim.Adam(self.solver.parameters(), lr=self.lr)

    def _train(self):
        self.solver.train()
        for i in range(1, self.num_epochs + 1):
            ep_loss = 0.0
            for j in range(1, self.batch_size + 1):
                self.optimizer.zero_grad()
                batch_data = torch.rand(
                    size=(self.num_instances_per_batch, self.num_nodes, 2),
                    device=self.device,
                    dtype=self.dtype,
                )
                sols, log_probs = self.solver(batch_data, greedy=False)
                with torch.no_grad():
                    self.greedy.eval()
                    greedy_sols, _ = self.greedy(batch_data, greedy=True)
                    dist1 = cost(batch_data, sols)
                    dist2 = cost(batch_data, greedy_sols)

                loss = (dist1 - dist2) * log_probs
                loss = torch.mean(loss)

                ep_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.solver.parameters(), 0.25)
                self.optimizer.step()
                if not j % 10:
                    print(
                        "<TRAIN> EPOCH: {:d}/{:d} BATCH: {:d}/{:d} LOSS: {:>7f}".format(
                            i,
                            self.num_epochs,
                            j,
                            self.batch_size,
                            loss.item(),
                        )
                    )

            self.loss_history_train.append(ep_loss / self.batch_size)
            print("<END TRAIN> AVG LOSS: {:>7f}".format(ep_loss / self.batch_size))

            self._improve()
            if not i % args.save_interval:
                torch.save(
                    {
                        "loss_history": self.loss_history_train,
                        "model": self.solver.state_dict(),
                        "optim": self.optimizer.state_dict(),
                    },
                    f="results/checkpoints/checkpoint_{}.pkl".format(i),
                )

    @torch.no_grad()
    def _improve(self):
        self.solver.eval()
        self.greedy.eval()
        x = torch.rand(
            size=(self.batch_size_val, self.num_nodes, 2),
            device=self.device,
            dtype=self.dtype,
        )
        s1, _ = self.solver(x, greedy=False)
        s2, _ = self.greedy(x, greedy=True)
        sols1 = np.concatenate([cost(x, s1).cpu().numpy()]).reshape(-1)
        sols2 = np.concatenate([cost(x, s2).cpu().numpy()]).reshape(-1)
        _, p_value = stats.ttest_rel(sols1, sols2)
        loss = (sols1 - sols2).mean()
        improve = loss >= 0
        if improve and p_value <= 0.05:
            self.greedy.load_state_dict(self.solver.state_dict())

        print("<TEST> AVG LOSS: {:>7f}".format(loss))
        self.loss_history_test.append(loss)


if __name__ == "__main__":
    with open("args.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
        args = SimpleNamespace(**args)

    solver = TSPSolver(args)
    solver._train()
    plt.plot(solver.loss_history_train)
    plt.show()
