import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
print("BACKEND: ", matplotlib.get_backend())
from matplotlib import pyplot as plt
import utility as ut
import network as nt
from tqdm import tqdm as tqdm
import plot as pt
from sklearn.decomposition import PCA

delta_T = 1e-2

# mnist
labels = [0, 3]
(x_train, y_train), (x_test, y_test) = ut.mnist.load_data()
selection = [y_test == label for label in labels]

minimum_length = min(np.sum(selection, axis=1))
selection = np.any([np.all((item, np.cumsum(item) < minimum_length), axis=0) for item in selection], axis=0)
X = x_test[selection]
Y = y_test[selection]
X = X.reshape((len(X), -1)) / 255.0
X = (X > 0.5).astype(np.float32)
X_frequencies = X * 70.0 + 20.0

for label in labels:
    print(sum(Y==label))

X_spikes = ut.generate_spike_trains(X_frequencies, 10000, delta_T=delta_T, batch_size=1)

n_outputs = 12
n_inputs = 28*28
r_net = 20.0 # 0.5
m_k = 1.0/n_outputs

net = nt.BinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                            delta_T=delta_T, r_net=r_net, m_k=m_k, eta_v=1e0, eta_b=1e1)

fig = plt.figure(figsize=(3.5, 1.16), dpi=300)
#plt.show(block=False)
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

    imshows.append(ax.imshow(ut.sigmoid(net._V[i].reshape((28, 28))), vmin=0.3, vmax=0.7))
fig.canvas.draw()
fig.canvas.flush_events()

# set up figure for PCA
fig_pca, ax_pca = plt.subplots(1)
colors = ["C0", "C1", "C2", "C3"]
Y_color = np.empty(Y.shape, dtype="object")
for i, label in enumerate(labels):
    Y_color[Y == label] = colors[i%len(colors)]
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

scatter_constant = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_color,alpha=0.5, marker="o", s=2)
scatter_variable = ax_pca.scatter(np.zeros(n_outputs), np.zeros(n_outputs), c="black", marker="o", s=4)
ax_pca.legend()
plt.show(block=False)

# train
pbar = tqdm(enumerate(X_spikes))
for batch_index, sample_batch in pbar:
    # update figure here
    for sample in sample_batch:
        net.step(sample)

    # update figures every percent
    #if not batch_index % 1:
    # reshape to 28x28 to plot
    weights = net._V.reshape((-1, 28, 28))
    weights_pca = pca.transform(ut.sigmoid(net._V))
    scatter_variable.set_offsets(weights_pca)
    fig_pca.canvas.draw()
    fig_pca.canvas.flush_events()
    # scatter_variable = ax_pca.scatter(weights_pca[:, 0], weights_pca[:, 1], c="black", marker="o")
    for i in range(len(imshows)):
        # pi_k_i = sigmoid(weight)
        imshows[i].set_data(ut.sigmoid(weights[i]))

    fig.canvas.draw()
    pbar.set_description(f'<|V|> = {np.mean(np.abs(net._V)):.4f}, <b> = {np.mean(net._b):.4f}')
    fig.canvas.flush_events()


