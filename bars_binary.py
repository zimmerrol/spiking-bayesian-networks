import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
print("BACKEND: ", matplotlib.get_backend())
from matplotlib import pyplot as plt
import utility as ut
import network as nt
from tqdm import tqdm as tqdm
import plot as pt

delta_T = 1e-3

# bars
spiking_input = False
dim = 8
n_outputs = 2*dim
n_inputs = dim*dim
r_net = 2 # 0.5
m_k = 1.0/n_outputs
X = ut.generate_bars(10000, dim, dim, p=1.7/8.0)

X = np.reshape(X, (-1, dim*dim))
if spiking_input:
    X = X * 70.0 + 20.0
    X_spikes = ut.generate_spike_trains(X, 1000, delta_T=delta_T)
else:
    X_spikes = ut.generate_constant_trains(X, 1000, delta_T=delta_T)


"""
# visualize spike trains
test_spikes = list(X_spikes)[0]
pt.plot_spiketrain(test_spikes, delta_T, tmax=2)
plt.show()
"""

net = nt.BinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                            delta_T=delta_T, r_net=r_net, m_k=m_k, eta_v=1e2, eta_b=1e5)



# train
from plot import WeightPCAPlotter, WeightPlotter
pca_plotter = WeightPCAPlotter(X, np.zeros(X.shape[0]), n_outputs, [0, 0], annotations=True)
weights_plotter = WeightPlotter(ut.sigmoid(net._V).reshape((-1, dim, dim)))

from collections import deque

average_length_likelihood = 500

pbar = tqdm(enumerate(X_spikes))
for batch_index, sample_batch in pbar:
    # update figure here
    log_likelihoods = deque([])
    for sample in sample_batch:
        net.step(sample)

        # log likelihood
        Ak = np.sum(np.log(1+np.exp(net._V)), -1)

        pi = ut.sigmoid(net._V)
        log_likelihoods.append(np.log(1.0/n_outputs) + np.log(np.sum(np.prod(sample * pi + (1-sample) * (1-pi), axis=-1))))

        if len(log_likelihoods) > average_length_likelihood:
            log_likelihoods.popleft()

    weights = ut.sigmoid(net._V)

    pca_plotter.update(weights)

    weights_plotter.update(weights)

    pbar.set_description(f'<sigma(V)> = {np.mean(weights):.4f}, <b> = {np.mean(net._b):.4f}, <L(y)> = {np.mean(log_likelihoods)}')

