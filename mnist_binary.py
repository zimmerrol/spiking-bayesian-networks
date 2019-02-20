import numpy as np
import utility as ut
import network as nt
from tqdm import tqdm as tqdm
from plot import WeightPCAPlotter, WeightPlotter, CurvePlotter
from collections import deque

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


net = nt.EventBasedBinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                            r_net=r_net, m_k=m_k, eta_v=1e-2, eta_b=1e+0, max_trace_length=1000)


# train

pca_plotter = WeightPCAPlotter(X, Y, n_outputs, labels)
weights_plotter = WeightPlotter(ut.sigmoid(net._V).reshape((-1, W, H)))
likelihood_plotter = CurvePlotter(x_label="Time [s]", y_label="$log(p(y))$")

average_length_likelihood = 500

likelihoods = []

log_likelihoods = deque([])

T_max = 100

pbar = tqdm(total=T_max, unit='Time [s]')
while net._current_time < T_max:
    net.step(data_generator)

    pbar.n = int(net._current_time * 1000) / 1000
    pbar.update(0)

    # log likelihood

    sample = net._trace[-1][1].reshape((1, -1))
    Ak = np.sum(np.log(1+np.exp(net._V)), -1)

    pi = ut.sigmoid(net._V)
    log_likelihoods.append(np.log(1.0/n_outputs) + np.log(np.sum(np.prod(sample * pi + (1-sample) * (1-pi), axis=-1))))

    if len(log_likelihoods) > average_length_likelihood:
        log_likelihoods.popleft()

    # update plots
    if int(pbar.n) > len(likelihoods):
        likelihoods.append(np.mean(log_likelihoods))

        weights_plotter.update(ut.sigmoid(net._V))
        pca_plotter.update(ut.sigmoid(net._V))
        likelihood_plotter.update(likelihoods)

    pbar.set_description(
        f'<sigma(V)> = {np.mean(ut.sigmoid(net._V)):.4f}, <b> = {np.mean(net._b):.4f}, <L(y)> = {np.mean(log_likelihoods):.4f}')

pbar.close()
