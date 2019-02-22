"""
Train a spiking Bayesian WTA network.
Uses a model that assumes non.binary causes/outputs but rather EPSP like outputs.

MIT License

Copyright (c) 2019 Roland Zimmermann, Laurenz Hemmen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import utility as ut
import network as nt
from tqdm import tqdm as tqdm
from collections import deque
from copy import deepcopy
from data_generator import DataGenerator

delta_t = 1e-3

# parameters
spiking_input = False
labels = [0, 3]
n_outputs = 12
W, H = 24, 24
r_net = 50.0
t_max = 1000

n_inputs = W*H
m_k = 1.0/n_outputs

# load data
x, y = ut.load_mnist(h=H, w=W, labels=labels, train=False, frequencies=spiking_input)


def estimate_likelihood(estimation_duration=100.0):
    log_likelihoods = deque([])

    estimation_net = deepcopy(net)
    estimation_net._current_time = 0
    estimation_net._trace = deque([])

    while estimation_net._current_time < estimation_duration:
        z = estimation_net.step(lambda t: data_generator[t], update_weights=False)

        pbar.n = int(net._current_time * 1000) / 1000
        pbar.update(0)

        # log likelihood
        sample = estimation_net._trace[-1][1].reshape((-1, 1))

        norm = np.sum(1 + np.exp(np.dot(z.reshape((1, -1)), net._V)))
        log_likelihoods.append(-norm + np.sum(z * np.dot(net._V, sample), axis=-1))

    return np.mean(log_likelihoods), np.std(log_likelihoods)


likelihoods = {}
tau_pbar = tqdm(10**np.linspace(-4, -2, 3), position=0)
for tau in tau_pbar:
    net = nt.EventBasedOutputEPSPBinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                                                  r_net=r_net, m_k=m_k,eta_v=1e-2, eta_b=1e+0,
                                                  max_trace_length=1000, tau=tau, delta_t=tau/10)

    data_generator = DataGenerator(x, 10000, t_image=0.250, delta_t=delta_t)

    pbar = tqdm(total=t_max, unit='Time [s]', leave=False, position=1)
    while net._current_time < t_max:
        z = net.step(lambda t: data_generator[t])

        pbar.n = int(net._current_time * 1000) / 1000
        pbar.update(0)

    likelihood = estimate_likelihood()
    likelihoods[tau] = likelihood

    pbar.close()

likelihoods = [[key, likelihoods[key][0], likelihoods[key][1]] for key in likelihoods]
likelihoods = np.array(likelihoods)

from matplotlib import pyplot as plt
plt.plot(likelihoods[:, 0], likelihoods[:, 1])
plt.fill_between(likelihoods[:, 0], likelihoods[:, 1]-likelihoods[:, 2], likelihoods[:, 1]+ likelihoods[:, 2], alpha=0.5)
plt.xlabel("$\\tau$")
plt.ylabel("$log[p(y)]$")
plt.show()
