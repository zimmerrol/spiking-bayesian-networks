"""
Evaluates the reconstruction quality for varying number of output neurons.

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


def estimate_likelihood(estimation_duration=10.0):
    log_likelihoods = deque([])

    estimation_net = deepcopy(net)
    estimation_net._current_time = 0
    estimation_net._trace = deque([])

    while estimation_net._current_time < estimation_duration:
        estimation_net.step(lambda t: data_generator[t], update_weights=False)

        pbar.n = int(net._current_time * 1000) / 1000
        pbar.update(0)

        # log likelihood
        y = estimation_net._trace[-1][1].reshape((1, -1))

        pi = ut.sigmoid(net._V)
        log_likelihoods.append(
            np.log(1.0 / n_outputs) + np.log(np.sum(np.prod(y * pi + (1 - y) * (1 - pi), axis=-1))))

    return np.mean(log_likelihoods), np.std(log_likelihoods)


def reconstruct(net, input, t_image=0.250):
    estimation_net = deepcopy(net)
    estimation_net._current_time = 0
    estimation_net._trace = deque([])

    reconstruction = np.zeros_like(input)
    while estimation_net._current_time < t_image:
        z = estimation_net.step(lambda t: input, update_weights=False)

        reconstruction += z.dot(ut.sigmoid(net._V))

    return reconstruction


def reconstruction_likelihood(net, t_image = 0.250):
    spikes = np.zeros((len(x), n_outputs))

    pi = ut.sigmoid(net._V)

    likelihoods =[]

    estimation_net = deepcopy(net)
    estimation_net._current_time = 0
    estimation_net._trace = deque([])

    pbar = tqdm(total=len(x) * t_image, unit='Time [s]', position=1, desc="Reconstruction")
    while estimation_net._current_time < len(x) * t_image:
        pbar.n = int(estimation_net._current_time * 1000) / 1000
        pbar.update(0)

        z = estimation_net.step(lambda t: x[int(min(t, (len(x)-1) * t_image) / t_image)], update_weights=False)

        sample = x[int(min(estimation_net._current_time, (len(x)-1) * t_image) / t_image)]
        pi = ut.sigmoid(np.dot(z.reshape((1, -1)), net._V))
        likelihoods.append(np.sum(np.log(sample * pi + (1 - sample) * (1 - pi)), axis=-1))

    pbar.close()

    return np.mean(likelihoods)


def reconstruction_l2_loss(net, t_image=0.250):
    estimation_net = deepcopy(net)
    estimation_net._current_time = 0
    estimation_net._trace = deque([])

    spikes = np.zeros((len(x), n_outputs))
    pbar = tqdm(total=len(x) * t_image, unit='Time [s]', position=1, desc="Reconstruction")
    while estimation_net._current_time < len(x) * t_image:
        pbar.n = int(estimation_net._current_time * 1000) / 1000
        pbar.update(0)

        z = estimation_net.step(lambda t: x[int(min(t, (len(x) - 1) * t_image) / t_image)], update_weights=False)
        spikes[min(len(x)-1, int(estimation_net._current_time / t_image))] += z.flatten()

    reconstructions = np.dot(spikes, ut.sigmoid(estimation_net._V)) / np.sum(spikes, axis=-1).reshape(-1, 1)
    difference = np.mean((reconstructions - x) ** 2)

    return difference


losses = {}
for n_outputs in tqdm(range(2, 24)):
    m_k = 1.0 / n_outputs
    net = nt.EventBasedBinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                                        r_net=r_net, m_k=m_k, eta_v=1e-2, eta_b=1e+0, max_trace_length=1000)

    data_generator = DataGenerator(x, t_max*10, t_image=0.250, delta_T=delta_T, spiking=spiking_input)
    pbar = tqdm(total=t_max, unit='Time [s]')
    while net._current_time < t_max:
        z = net.step(lambda t: data_generator[t])

        pbar.n = int(net._current_time * 1000) / 1000
        pbar.update(0)
        pbar.set_description(f'<sigma(V)> = {np.mean(ut.sigmoid(net._V)):.4f}, <b> = {np.mean(net._b):.4f}')

    losses[n_outputs] = reconstruction_likelihood(net)
    # losses[n_outputs] = reconstruction_l2_loss(net)

losses = np.array([[key, losses[key]] for key in losses])

np.save("reconstruction_likelihood_loss.npy", losses)
# np.save("reconstruction_l2_loss.npy", losses)

from matplotlib import pyplot as plt
plt.plot(losses[:, 0], losses[:, 1])
plt.show()
