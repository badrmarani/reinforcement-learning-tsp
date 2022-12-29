# Solving TSP using Reinforcement Learning and Optimal Transport

This repository contains a PyTorch implementation of the paper: [Combining Reinforcement Learning and Optimal Transport for the Traveling Salesman Problem](https://arxiv.org/abs/2203.00903).

## Requirements

```sh
python -m pip install -r requirements.txt
```

## Training a new model

First change the parameters on `args.yml` and then run the following command:

```sh
python train.py
```

Once the model is trained, run:

```sh
python -m jupyter nbconvert --to python demo.ipynb
python demo.py
```

## References

- Goh, Yong Liang, et al. Combining Reinforcement Learning and Optimal Transport for the Traveling Salesman Problem. arXiv, 2 mars 2022. arXiv.org, [https://doi.org/10.48550/arXiv.2203.00903](https://doi.org/10.48550/arXiv.2203.00903).
- Kool, Wouter, et al. Attention, Learn to Solve Routing Problems! arXiv, 7 février 2019. arXiv.org, [https://doi.org/10.48550/arXiv.1803.08475](https://doi.org/10.48550/arXiv.1803.08475).
- Bresson, Xavier, et Thomas Laurent. The Transformer Network for the Traveling Salesman Problem. arXiv, 4 mars 2021. arXiv.org, [https://doi.org/10.48550/arXiv.2103.03012](https://doi.org/10.48550/arXiv.2103.03012).
