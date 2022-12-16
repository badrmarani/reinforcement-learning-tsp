from scipy import stats
import numpy as np
import torch


def cost(graphs, permutations):
    """copied from https://github.com/wouterkool/attention-learn-to-route"""
    d = graphs.gather(1, permutations.unsqueeze(-1).expand_as(graphs))
    return (
        (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
        + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)
    ).view(-1, 1)


def draw(x, solution, ax):
    ax.plot(x[0, solution[0, :], 0], x[0, solution[0, :], 1], "b-o")
    ax.plot(
        [x[0, solution[0, 0], 0], x[0, solution[0, -1], 0]],
        [x[0, solution[0, 0], 1], x[0, solution[0, -1], 1]],
        "b-o",
    )
    ax.axis(False)


class Trainer:
    def __init__(self, args, dtype, device) -> None:
        self.num_batchs = args["num_batchs"]
        self.num_instances_per_batchs = args["num_instances_per_batchs"]
        self.num_nodes = args["num_nodes"]
        self.embed_size = args["embed_size"]
        self.num_heads = args["num_heads"]
        self.device = device
        self.dtype = dtype

    def train(self, epoch, tot_epochs, models, optim):
        model, greedy = models
        model.train()
        train_loss = 0.0
        for batch in range(self.num_batchs):
            x = torch.rand(
                size=(self.num_instances_per_batchs, self.num_nodes, 2),
                device=self.device,
                dtype=self.dtype,
            )
            solutions, lprobs = model(x)
            with torch.no_grad():
                greedy.eval()
                c1 = cost(x, solutions)
                greedy_solutions, _ = greedy(x)
                c2 = cost(x, greedy_solutions)

            loss = (c1 - c2) * lprobs
            loss = torch.mean(loss)

            optim.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()

            train_loss += loss.item()
            if not batch % 10:
                print(
                    "<TRAIN> EPOCH: {:d}/{:d} BATCH: {:d}/{:d} LOSS: {:>7f}".format(
                        epoch,
                        tot_epochs,
                        batch * self.num_instances_per_batchs,
                        self.num_batchs * self.num_instances_per_batchs,
                        loss.item(),
                    )
                )

        train_loss /= self.num_batchs
        print(
            "<TRAIN ERROR> EPOCH: {:d}/{:d} AVG LOSS: {:>7f}".format(
                epoch, tot_epochs, train_loss
            )
        )
        return model, greedy, train_loss

    @torch.no_grad()
    def test(self, epoch, tot_epochs, batch_size_val, models, optim):
        model, greedy = models
        model.eval()
        greedy.eval()

        x = torch.rand(
            size=(batch_size_val, self.num_nodes, 2),
            device=self.device,
            dtype=self.dtype,
        )
        s1, _ = model(x)
        s2, _ = greedy(x)
        sols1 = np.concatenate([cost(x, s1).cpu().numpy()]).reshape(-1)
        sols2 = np.concatenate([cost(x, s2).cpu().numpy()]).reshape(-1)
        _, p_value = stats.ttest_rel(sols1, sols2)
        loss = (sols1 - sols2).mean()
        improve = loss >= 0
        if improve and p_value <= 0.05:
            greedy.load_state_dict(model.state_dict())

        print(
            "<TEST ERROR> EPOCH: {:d}/{:d} AVG LOSS: {:>7f}".format(
                epoch, tot_epochs, loss
            )
        )
        return model, greedy, loss
