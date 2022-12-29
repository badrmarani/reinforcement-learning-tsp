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
