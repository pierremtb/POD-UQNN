# Uncertainty Quantification in the POD-NN framework

Source code from _Non-intrusive reduced-order modeling using uncertainty-aware Deep Neural Networks and Proper Orthogonal Decomposition: application to Flood Modeling_.

Requires Python 3.6+.
Dependencies are in the file `requirements.txt` on any branch, and are installable via `pip` (or `pip3` if Python 3 isn’t the default one):
```console
$ pip3 install -r requirements.txt
```

## Running our POD-NN implementation (Wang et al., 2019)
Run  experiments from their directories, eg.
```console
$ git checkout POD-NN
$ cd  experiments/1d_shekel
$ python3 main.py
```
Available experiments in ` experiments`:
- `1d_shekel`, the 1D [Shekel](https://en.wikipedia.org/wiki/Shekel_function) function
- `2d_ackley`, the 2D [Ackley](https://en.wikipedia.org/wiki/Ackley_function) function
- `1dt_burger`, a solution of the 1D, unsteady [Burger’s Equation](https://en.wikipedia.org/wiki/Burgers%27_equation)
- `2d_shallowwater`, Flood Modeling on simulations results from CuteFlow, solving 2D [Shallow Water Equations](https://en.wikipedia.org/wiki/Shallow_water_equations)

## Running the POD-EnsNN model (Uncertainty Quantification via Deep Ensembles)
```console
$ git checkout POD-EnsNN
$ cd  experiments/1d_shekel
$ python3 gen.py && python3 train.py --models=5 && python3 pred.py
```
Or to distribute on a machine with 5 GPUs
```console
$ python3 gen.py && horovodrun -np 5 -H localhost:5 python3 train.py --distribute && python3 pred.py
```
Available experiments in ` experiments`:
- `1d_shekel`, the 1D [Shekel](https://en.wikipedia.org/wiki/Shekel_function) function
- `2d_ackley`, the 2D [Ackley](https://en.wikipedia.org/wiki/Ackley_function) function
- `1dt_burger`, a solution of the 1D, unsteady [Burger’s Equation](https://en.wikipedia.org/wiki/Burgers%27_equation)
- `2d_shallowwater`, Flood Modeling on simulations results from CuteFlow, solving 2D [Shallow Water Equations](https://en.wikipedia.org/wiki/Shallow_water_equations)
- `2dt_shallowwater`, Dam Break simulations results from CuteFlow, solving 2D, unsteady [Shallow Water Equations](https://en.wikipedia.org/wiki/Shallow_water_equations)

## Running the POD-BNN model (Uncertainty Quantification via Bayesian NN)
```console
$ git checkout POD-BNN
$ cd  experiments/1d_shekel
$ python3 main.py
```
Available experiments in ` experiments`:
- `1d_shekel`, the 1D [Shekel](https://en.wikipedia.org/wiki/Shekel_function) function
- `2d_ackley`, the 2D [Ackley](https://en.wikipedia.org/wiki/Ackley_function) function
- `1dt_burger`, a solution of the 1D, unsteady [Burger’s Equation](https://en.wikipedia.org/wiki/Burgers%27_equation)
- `2d_shallowwater`: Flood Modeling on simulations results from CuteFlow, solving 2D [Shallow Water Equations](https://en.wikipedia.org/wiki/Shallow_water_equations)

## Runner files for Compute Canada clusters
For each branch, we provide `experiments/runner.sh` to run all simulations. It is meant to be used on Compute Canada clusters, such as Beluga, located at ÉTS.
A Python 3.6+ environment at `~/env` needs to contain the packages required in `requirements.txt`, plus `horovod`.

## Citation
This work is building on techniques from _Wang et al._
```
@article{Wang2019,
author = {Wang, Qian and Hesthaven, Jan S. and Ray, Deep},
doi = {10.1016/J.JCP.2019.01.031},
issn = {0021-9991},
journal = {Journal of Computational Physics},
month = {may},
pages = {289--307},
publisher = {Academic Press},
title = {{Non-intrusive reduced order modeling of unsteady flows using artificial neural networks with application to a combustion problem}},
url = {https://www.sciencedirect.com/science/article/pii/S0021999119300828},
volume = {384},
year = {2019}
}
```

## License
MIT License

Copyright (c) 2019 Pierre Jacquier
