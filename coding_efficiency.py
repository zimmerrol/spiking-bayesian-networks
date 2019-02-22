"""
Measure the sparsity of the output code of a spiking network for varying number of output nodes.

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


def calculate_coding_efficiency(net, t_image=0.250):
    estimation_net = deepcopy(net)
    estimation_net._current_time = 0
    estimation_net._trace = deque([])

    spikes = np.zeros((len(x), n_outputs))
    pbar = tqdm(total=len(x) * t_image, unit='Time [s]', position=1, desc="Reconstruction")
    while estimation_net._current_time < len(X) * t_image:
        pbar.n = int(estimation_net._current_time * 1000) / 1000
        pbar.update(0)

        z = estimation_net.step(lambda t: x[int(min(t, (len(x) - 1) * t_image) / t_image)], update_weights=False)
        spikes[min(len(x)-1, int(estimation_net._current_time / t_image))] += z.flatten()

    max_spikes = np.max(spikes, axis=-1)
    total_spikes = np.sum(spikes, axis=-1)

    return np.mean(max_spikes / total_spikes)


values = {}
for n_outputs in tqdm(range(2, 25)):
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

    values[n_outputs] = calculate_coding_efficiency(net)

values = np.array([[key, values[key]] for key in values])
np.save("coding_efficiencies.npy", values)

from matplotlib import pyplot as plt
plt.plot(values[:, 0], values[:, 1])
plt.show()
