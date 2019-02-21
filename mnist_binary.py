import numpy as np
import utility as ut
import network as nt
from tqdm import tqdm as tqdm
from plot import WeightPCAPlotter, WeightPlotter, CurvePlotter, SpiketrainPlotter
from collections import deque
from  copy import deepcopy

delta_T = 1e-3

# mnist
spiking_input = False
labels = [2, 4]
n_outputs = 12
W, H = 24, 24
n_inputs = W*H
r_net = 50.0 # 0.5
m_k = 1.0/n_outputs



(x_train, y_train), (x_test, y_test) = ut.mnist.load_data()
selection = [y_test == label for label in labels]

minimum_length = min(np.sum(selection, axis=1))
selection = np.any([np.all((item, np.cumsum(item) < minimum_length), axis=0) for item in selection], axis=0)
X = x_test[selection]

X = X[:, (28-H)//2:-(28-H)//2, (28-W)//2:-(28-W)//2]

Y = y_test[selection]
X = X.reshape((len(X), -1)) / 255.0
X = (X > 0.5).astype(np.float32)
X_frequencies = X * 70.0 + 20.0


def data_generator(time):
    image_index = np.random.uniform(0, len(X), 1).astype(np.int32)

    if spiking_input:
        rate = X_frequencies[image_index]
        y = (np.random.uniform(0, 1, n_inputs) < delta_T * rate).astype(np.uint8)
    else:
        y = X[image_index]

    return y


net = nt.EventBasedOutputEPSPBinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                            r_net=r_net, m_k=m_k, eta_v=1e-2, eta_b=1e+0, max_trace_length=1000,
                            tau=1e-2, delta_T=1e-3)


# train

pca_plotter = WeightPCAPlotter(X, Y, n_outputs, labels)
weights_plotter = WeightPlotter(ut.sigmoid(net._V).reshape((-1, W, H)))
likelihood_plotter = CurvePlotter(x_label="Time [s]", y_label="$log(p(y))$")
output_spiketrains = None
# uncomment to plot the spiketrains
# output_spiketrains = SpiketrainPlotter(n_outputs, 100)

likelihoods = []

T_max = 100


def estimate_likelihood(estimation_duration=4.0):
    log_likelihoods = deque([])

    estimation_net = deepcopy(net)
    estimation_net._current_time = 0
    estimation_net._trace = deque([])

    while estimation_net._current_time < estimation_duration:
        estimation_net.step(data_generator, update_weights=False)

        pbar.n = int(net._current_time * 1000) / 1000
        pbar.update(0)

        # log likelihood

        sample = estimation_net._trace[-1][1].reshape((1, -1))

        pi = ut.sigmoid(net._V)
        log_likelihoods.append(
            np.log(1.0 / n_outputs) + np.log(np.sum(np.prod(sample * pi + (1 - sample) * (1 - pi), axis=-1))))

    return np.mean(log_likelihoods), np.std(log_likelihoods)


pbar = tqdm(total=T_max, unit='Time [s]')
while net._current_time < T_max:
    z = net.step(data_generator)

    if output_spiketrains is not None:
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
