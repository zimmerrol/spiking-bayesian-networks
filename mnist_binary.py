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

if spiking_input:
    X_frequencies = X * 70.0 + 20.0
    X_spikes = ut.generate_spike_trains(X_frequencies, 1000, delta_T=delta_T)
else:
    X_spikes = ut.generate_constant_trains(X, 1000, delta_T=delta_T)

net = nt.BinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                            delta_T=delta_T, r_net=r_net, m_k=m_k, eta_v=1e+1, eta_b=1e+3)
    

# train
from plot import WeightPCAPlotter, WeightPlotter
pca_plotter = WeightPCAPlotter(X, Y, n_outputs, labels)
weights_plotter = WeightPlotter()

pbar = tqdm(enumerate(X_spikes))
for batch_index, sample_batch in pbar:
    # update figure here
    for sample in sample_batch:
        net.step(sample)

    weights = ut.sigmoid(net._V)

    pca_plotter.update(weights)
    weights_plotter.update(weights)

    pbar.set_description(f'<sigma(V)> = {np.mean(weights):.4f}, <b> = {np.mean(net._b):.4f}')
    

# log likelihood
# 