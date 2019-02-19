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
dim = 8
X = ut.generate_bars(10000, dim, dim, p=1.7/8.0)

X = np.reshape(X, (-1, dim*dim))
X_frequencies = X * 70.0 + 20.0


X_spikes = ut.generate_spike_trains(X_frequencies, 1000, delta_T=delta_T)

#pt.plot_spiketrain(X_spikes, delta_T)
#plt.show()


n_outputs = 2*dim
n_inputs = dim*dim
r_net = 2 # 0.5
m_k = 1.0/n_outputs

net = nt.BinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                            delta_T=delta_T, r_net=r_net, m_k=m_k, eta_v=1e0, eta_b=1e-0)

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

    imshows.append(ax.imshow(ut.sigmoid(net._V[i].reshape((dim, dim))), vmin=0, vmax=1))
fig.canvas.draw()
fig.canvas.flush_events()

# train
pbar = tqdm(range(len(X_spikes)))
for i in pbar:
    # update figure here
    net.step(X_spikes[i])

    # update figures every percent
    if not i % int(100 * (0.25 / delta_T)):
        # reshape to 28x28 to plot
        weights = net._V.reshape((-1, dim, dim))
        for i in range(len(imshows)):
            # pi_k_i = sigmoid(weight)
            imshows[i].set_data(ut.sigmoid(weights[i]))

        fig.canvas.draw()
    pbar.set_description(f'<|V|> = {np.mean(np.abs(net._V)):.4f}, <|b|> = {np.mean(np.abs(net._b)):.4f}')
    fig.canvas.flush_events()
