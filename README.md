Implementation of the paper [Homeostatic plasticity in Bayesian spiking networks as Expectation Maximization with posterior constraints](https://papers.nips.cc/paper/4593-homeostatic-plasticity-in-bayesian-spiking-networks-as-expectation-maximization-with-posterior-constraints) by Habenschuss et al.

Contains code to runs different experiments on the proposed model and also on a model that is based not on a Binomial but Gaussian input distribution.

# Key insights
* `eta_b` has to be sufficiently large, otherwise homeostasis is not stronk enough to keep `r` similar for all output neurons
* `A_k(V)` contributes exponentially `b_k` linearly, it is not always factor ten between `eta_V` and `eta_b`
*  Too few neurons for causes lead to learning of superposition states
* Network can reconstruct images is was not trained on