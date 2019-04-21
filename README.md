Implementation of the paper [Homeostatic plasticity in Bayesian spiking networks as Expectation Maximization with posterior constraints](https://papers.nips.cc/paper/4593-homeostatic-plasticity-in-bayesian-spiking-networks-as-expectation-maximization-with-posterior-constraints) by Habenschuss et al. This paper gives learning rules for a spiking neural network just based on Bayesian reasoning; therefore, the method can be used for unsupervised training of networks.

Contains code to runs different experiments on the proposed model and also on a model that is based not on a Binomial but Gaussian input distribution.

The code was written and the experiments conducted during a one week lasting seminar at the Max-Planck Institute for Dynamics and Self-Organization in 2019.

# Key insights
* `eta_b` has to be sufficiently large, otherwise homeostasis is not stronk enough to keep `r` similar for all output neurons
* `A_k(V)` contributes exponentially `b_k` linearly, it is not always factor ten between `eta_V` and `eta_b`
*  Too few neurons for causes lead to learning of superposition states
* Network can reconstruct images is was not trained on

When images of digits between zero and five with the same ratio are shown to a network with 12 output neurons, for each class two neurons that are class-receptive arise. The neurons slowly learn to react to one of the input types.
![Visualization of the learning process](weights.gif)

![Visualization of the learning process](weights_pca.gif)
