import numpy as np
import utility as ut
import network as nt
from tqdm import tqdm as tqdm
from plot import WeightPCAPlotter, WeightPlotter, CurvePlotter, SpiketrainPlotter
from collections import deque
from  copy import deepcopy

delta_T = 1e-3

# mnist
spiking_input = True
labels = [2, 4]
ratio = [1, 1]
n_outputs = 12
W, H = 24, 24
n_inputs = W*H
r_net = 50.0 # 0.5
m_k = 1.0/n_outputs



(x_train, y_train), (x_test, y_test) = ut.mnist.load_data()
selection = [y_test == label for label in labels]

minimum_length = min(np.sum(selection, axis=1))
selection = np.any([np.all((item, np.cumsum(item) < minimum_length/ratio[i]), axis=0) for i,item in enumerate(selection)], axis=0)
X = x_test[selection]
Y = y_test[selection]
for label in labels:
	    print(sum(Y==label))

X = X[:, (28-H)//2:-(28-H)//2, (28-W)//2:-(28-W)//2]

Y = y_test[selection]
X = X.reshape((len(X), -1)) / 255.0
X = (X > 0.5).astype(np.float32)
X_frequencies = X * 70.0 + 20.0


class DataGenerator():
    def __init__(self, X, length, delta_T, t_image, spiking=False):
        self._X = X
        self._indices = np.random.uniform(0, len(X), size=length).astype(np.int32)
        self._delta_T = delta_T
        self._t_image = t_image
        self._spiking = spiking
        self._cache = SpikeTrainCache(delta_T)

    def __getitem__(self, time):
        if time in self._cache:
            return self._cache[time]

        image_index = self._indices[int(time / self._t_image)]

        if self._spiking:
            rate = self._X[image_index]
            y = (np.random.uniform(0, 1, self._X.shape[1]) < self._delta_T * rate).astype(np.uint8)
        else:
            y = self._X[image_index]

        self._cache[time] = y

        return y


class SpikeTrainCache():
    def __init__(self, delta_T):
        self._delta_T = delta_T
        self._trace = {}

    def _bin_time(self, time):
        return int(time / self._delta_T)

    def __contains__(self, time):
        return self._bin_time(time) in self._trace

    def __getitem__(self, time):
        return self._trace[self._bin_time(time)]

    def __setitem__(self, time, value):
        self._trace[self._bin_time(time)] = value


net = nt.EventBasedOutputEPSPBinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                            r_net=r_net, m_k=m_k, eta_v=1e-2, eta_b=1e+0, max_trace_length=1000,
                            tau=1e-2, delta_T=1e-3)


# train

pca_plotter = WeightPCAPlotter(X, Y, n_outputs, labels)
weights_plotter = WeightPlotter(ut.sigmoid(net._V).reshape((-1, W, H)))
likelihood_plotter = CurvePlotter(x_label="Time [s]", y_label="$log[p(y)]$")
output_spiketrains = None
# uncomment to plot the spiketrains
# output_spiketrains = SpiketrainPlotter(n_outputs, 100)

likelihoods = []

T_max = 100


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
pbar = tqdm(total=T_max, unit='Time [s]')
while net._current_time < T_max:
    z = net.step(lambda t: data_generator[t])

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
