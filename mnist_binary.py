"""
Train a spiking Bayesian WTA network and plot weight changes, spike trains and log-likelihood live.

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
from plot import WeightPCAPlotter, WeightPlotter, CurvePlotter, SpiketrainPlotter
from collections import deque
from copy import deepcopy
from data_generator import DataGenerator

delta_T = 1e-3

# parameters
spiking_input = False
labels = [0, 1, 2, 3]
n_outputs = 12
W, H = 24, 24
r_net = 50.0
t_max = 1000

n_inputs = W*H
m_k = 1.0/n_outputs

# load data
x, y = ut.load_mnist(h=H, w=W, labels=labels, train=False, frequencies=spiking_input)


net = nt.EventBasedBinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                                    r_net=r_net, m_k=m_k, eta_v=1e-2, eta_b=1e+0, max_trace_length=1000)

# train
pca_plotter = WeightPCAPlotter(x, y, n_outputs, labels)
weights_plotter = WeightPlotter(ut.sigmoid(net._V).reshape((-1, W, H)))
likelihood_plotter = CurvePlotter(x_label="Time [s]", y_label="$log[p(y)]$")
output_spiketrains = SpiketrainPlotter(n_outputs, 100)

likelihoods = []


def estimate_likelihood(estimation_duration=5.0):
    log_likelihoods = deque([])

    estimation_net = deepcopy(net)
    estimation_net._current_time = 0
    estimation_net._trace = deque([])

    while estimation_net._current_time < estimation_duration:
        estimation_net.step(lambda t: data_generator[t], update_weights=False)

        pbar.n = int(net._current_time * 1000) / 1000
        pbar.update(0)

        # log likelihood

        sample = estimation_net._trace[-1][1].reshape((1, -1))

        pi = ut.sigmoid(net._V)
        log_likelihoods.append(
            np.log(1.0 / n_outputs) + np.log(np.sum(np.prod(sample * pi + (1 - sample) * (1 - pi), axis=-1))))

    return np.mean(log_likelihoods), np.std(log_likelihoods)


data_generator = DataGenerator(X, 10000, t_image=0.250, delta_T=delta_T, spiking=spiking_input)
pbar = tqdm(total=t_max, unit='Time [s]')
while net._current_time < t_max:
    z = net.step(lambda t: data_generator[t])

    if output_spiketrains is not None and net._current_time > 100:
        output_spiketrains.update([z], [net._current_time])

    pbar.n = int(net._current_time * 1000) / 1000
    pbar.update(0)

    # update plots
    if int(pbar.n) > len(likelihoods):
        likelihoods.append(estimate_likelihood())

        weights_plotter.update(ut.sigmoid(net._V))
        pca_plotter.update(ut.sigmoid(net._V))
        likelihood_plotter.update(likelihoods)

    likelihood = likelihoods[-1][0] if len(likelihoods) > 0 else np.nan
    pbar.set_description(
        f'<sigma(V)> = {np.mean(ut.sigmoid(net._V)):.4f}, <b> = {np.mean(net._b):.4f}, <L(y)> = {likelihood:.4f}')

pbar.close()
