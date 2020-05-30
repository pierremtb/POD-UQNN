---
layout: default
---
# Non-Intrusive Reduced-Order Modeling Using Uncertainty-Aware Deep Neural Networks and Proper Orthogonal Decomposition: Application to Flood Modeling
### Authors
[Pierre Jacquier](https://pierrejacquier.com), Azzedine Abdedou, Vincent Delmas, [Azzeddine Soulaïmani](https://www.etsmtl.ca/en/research/professors/asoulaimani/)

### Abstract
Deep Learning research is advancing at a fantastic rate, and there is
much to gain from transferring this knowledge to older fields like
Computational Fluid Dynamics in practical engineering contexts. This
work compares state-of-the-art methods that address uncertainty
quantification in Deep Neural Networks, pushing forward the
reduced-order modeling approach of Proper Orthogonal
Decomposition-Neural Networks (POD-NN) with Deep Ensembles and
Variational Inference-based Bayesian Neural Networks on two-dimensional
problems in space. These are first tested on benchmark problems, and
then applied to a real-life application: flooding predictions in the
Mille Îles river in the Montreal, Quebec, Canada metropolitan area. Our
setup involves a set of input parameters, with a potentially noisy
distribution, and accumulates the simulation data resulting from these
parameters. The goal is to build a non-intrusive surrogate model that is
able to know when it doesn’t know, which is still an open research area
in Neural Networks (and in AI in general). With the help of this model,
probabilistic flooding maps are generated, aware of the model
uncertainty. These insights on the unknown are also utilized for an
uncertainty propagation task, allowing for flooded area predictions that
are broader and safer than those made with a regular
uncertainty-uninformed surrogate model. Our study of the time-dependent
and highly nonlinear case of a dam break is also presented. Both the
ensembles and the Bayesian approach lead to reliable results for
multiple smooth physical solutions, providing the correct warning when
going out-of-distribution. However, the former, referred to as
POD-EnsNN, proved much easier to implement and showed greater
flexibility than the latter in the case of discontinuities, where
standard algorithms may oscillate or fail to converge.

* * * * * *

## Reduced Basis Generation with Proper Orthogonal Decomposition

We start by defining $u$, our $\mathbb{R}^{D}$-valued function of
interest. Computing this function is costly, so only a finite number $S$ of
solutions called *snapshots* can be realized. In our applications, the spatial mesh of $N_D$ nodes is considered fixed
in time, and since it is known and defined upfront, so we consider the number of outputs $H=N_D\times D$, the total number of degrees of freedom
(DOFs) of the mesh.
The simulation data, obtained from computing the function $u$ with $S$
parameter sets $\bm{s}^{(i)}$, is stored in a matrix of snapshots
$\bm{U} = [u_D(\bm{s}^{(1)})|\ldots|u_D(\bm{s}^{(S)})] \in \mathbb{R}^{H \times S}$.
Proper Orthogonal Decomposition (POD) is used to build a Reduced-Order
Model (ROM) and produce a *low-rank approximation*, which will be much
more efficient to compute and use when rapid multi-query simulations are
required. With the snapshots method, a
reduced POD basis can be efficiently extracted in a finite-dimension
context. In our case, we begin with the $\bm{U}$ matrix, and use the
Singular Value Decomposition algorithm, to extract
$\bm{W} \in \mathbb{R}^{H \times H}$,
$\bm{Z} \in \mathbb{R}^{S \times S}$ and the $r$ descending-ordered
positive singular values matrix
$\bm{D} = \text{diag}(\xi_1, \xi_2, \ldots, \xi_r)$ such that

##
    \bm{U} = \bm{W} \begin{bmatrix} \bm{D} & 0 \\ 0 & 0 \end{bmatrix} \bm{Z}^\intercal.
##

For the finite truncation of the first $L$ modes, the following
criterion on the singular values is imposed, with a hyperparameter
$\epsilon$ given as

##
    \dfrac{\sum_{l=L+1}^{r} \xi_l^2}{\sum_{l=1}^{r} \xi_l^2} \leq \epsilon,
##
and then each mode vector $\bm{V}_j \in \mathbb{R}^{S}$ can be found
from $\bm{U}$ and the $j$-th column of $\bm{Z}$, $\bm{Z}_j$, with

##
    \bm{V}_j = \dfrac{1}{\xi_j} \bm{U} \bm{Z}_j,
##
so that we can
finally construct our POD mode matrix

##
    \bm{V} = \left[\bm{V}_1 | \ldots | \bm{V}_j | \ldots | \bm{V}_L\right] \in \mathbb{R}^{H \times L}.
##
To project to and from the low-rank approximation requires projection
coefficients; those *corresponding* to the matrix of snapshots are
obtained by the following

##
    \bm{v} = \bm{V}^\intercal \bm{U},
##
and then $\bm{U}_\textrm{POD}$,
the approximation of $\bm{U}$, can be projected back to the expanded
space:

##
\bm{U}_\textrm{POD} = \bm{V}\bm{V}^\intercal\bm{U} = \bm{V} \bm{v}.
##

* * * * * 

**Acknowledgements**

This research was enabled in part by funding from the National Sciences
and Engineering Research Council of Canada and Hydro-Québec; by bathymetry data from the
[Communauté métropolitaine de Montréal](https://cmm.qc.ca/); and by
computational support from [Calcul Québec](www.calculquebec.ca) and
[Compute Canada](www.computecanada.ca).

* * * * *

### Citation
```
@misc{jacquier2020nonintrusive,
    title={Non-Intrusive Reduced-Order Modeling Using Uncertainty-Aware Deep Neural Networks and Proper Orthogonal Decomposition: Application to Flood Modeling},
    author={Pierre Jacquier and Azzedine Abdedou and Vincent Delmas and Azzeddine Soulaimani},
    year={2020},
    eprint={2005.13506},
    archivePrefix={arXiv},
    primaryClass={physics.comp-ph}
}
```
