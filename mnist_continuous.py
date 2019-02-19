import numpy as np
from matplotlib import pyplot as plt
import utility as ut
import network as nt
from tqdm import tqdm as tqdm
import plot as pt

# mnist
labels = [0, 1]
(x_train, y_train), (x_test, y_test) = ut.mnist.load_data()
selection = np.any([y_test == label for label in labels], axis=0)
X = x_test[selection]
Y = y_test[selection]
X = X.reshape((len(X), -1)) / 255.0
# X = (X > 0.5).astype(np.float32)

X_frequencies = X * 70.0 + 20.0

X_spikes = ut.generate_spike_trains(X_frequencies, 1000, delta_T=1e-2)

n_outputs = 12
n_inputs = 28*28
r_net = 20.0
m_k = 1/n_outputs

net = nt.ContinuousWTANetwork(n_inputs, n_outputs, 1e-2, r_net, m_k, eta_v=1e-1, eta_b=1e-0, eta_beta=1e-2)

fig = plt.figure(figsize=(3.5, 1.16), dpi=300)
plt.show(block=False)
# fig, axes = plt.subplots(2, 6)
axes = pt.add_axes_as_grid(fig, 2, 6, m_xc=0.01, m_yc=0.01)

imshows = []
for i, ax in enumerate( list(axes.flatten()) ):
    # disable legends
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    imshows.append(ax.imshow(ut.sigmoid(net._V[i].reshape((28, 28)))))
fig.canvas.draw()
fig.canvas.flush_events()

# train
pbar = tqdm(range(len(X_spikes)))
for i in pbar:
    # update figure here
    net.step(X_spikes[i])

    # update figures every percent
    if not i % int(5 * len(X_spikes)/100):
        # reshape to 28x28 to plot
        weights = net._V.reshape((-1, 28, 28))
        for i in range(len(imshows)):
            # pi_k_i = sigmoid(weight)
            imshows[i].set_data(weights[i])

        fig.canvas.draw()

    pbar.set_description(f'<|V|> = {np.mean(np.abs(net._V)):.4f}, <|b|> = {np.mean(np.abs(net._b)):.4f}, <beta> = {np.mean(net._beta):.4f}')

    fig.canvas.flush_events()

