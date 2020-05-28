---
layout: default
---
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
able to know when it doesn't know, which is still an open research area
in Neural Networks (and in AI in general). With the help of this model,
probabilistic flooding maps are generated, aware of the model
uncertainty. These insights on the unknown are also utilized for an
uncertainty propagation task, allowing for flooded area predictions that
are broader and safer than those made with a regular
uncertainty-uninformed surrogate model. Our study of the time-dependent
and highly nonlinear case of a dam break is also presented. Both the
ensembles and the Bayesian approach lead to reliable results for
multiple smooth physical solutions, providing the correct warning when
going *out-of-distribution*. However, the former, referred to as
POD-EnsNN, proved much easier to implement and showed greater
flexibility than the latter in the case of discontinuities, where
standard algorithms may oscillate or fail to converge.

* * * * * *

### Non-Intrusive Reduced-Order Modeling Using Uncertainty-Aware Deep Neural Networks and Proper Orthogonal Decomposition: Application to Flood Modeling
...

**Acknowledgements**

This research was enabled in part by funding from the National Sciences
and Engineering Research Council of Canada and Hydro-Québec; by bathymetry data from the
[Communauté métropolitaine de Montréal](https://cmm.qc.ca/); and by
computational support from [Calcul Québec](www.calculquebec.ca) and
[Compute Canada](www.computecanada.ca).

* * * * *

## Citation
@misc{jacquier2020nonintrusive,
    title={Non-Intrusive Reduced-Order Modeling Using Uncertainty-Aware Deep Neural Networks and Proper Orthogonal Decomposition: Application to Flood Modeling},
    author={Pierre Jacquier and Azzedine Abdedou and Vincent Delmas and Azzeddine Soulaimani},
    year={2020},
    eprint={2005.13506},
    archivePrefix={arXiv},
    primaryClass={physics.comp-ph}
}
