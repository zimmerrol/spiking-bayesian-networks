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

delta_T = 1e-3

# mnist
spiking_input = False
labels = [2, 4]
(x_train, y_train), (x_test, y_test) = ut.mnist.load_data()
selection = [y_test == label for label in labels]

minimum_length = min(np.sum(selection, axis=1))
selection = np.any([np.all((item, np.cumsum(item) < minimum_length), axis=0) for item in selection], axis=0)
X = x_test[selection]

X = X[:, 2:-2, 2:-2]

Y = y_test[selection]
X = X.reshape((len(X), -1)) / 255.0
X = (X > 0.5).astype(np.float32)

if spiking_input:
    X_frequencies = X * 70.0 + 20.0
    X_spikes = ut.generate_spike_trains(X_frequencies, 1000, delta_T=delta_T)
else:
    X_spikes = ut.generate_constant_trains(X, 1000, delta_T=delta_T)

n_outputs = 12
W, H = 24, 24
n_inputs = 24*24
r_net = 50.0 # 0.5
m_k = 1.0/n_outputs

net = nt.BinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
                            delta_T=delta_T, r_net=r_net, m_k=m_k, eta_v=1e+1, eta_b=1e+3)


class WeightPlotter():
    def __init__(self):
        self._fig = plt.figure(figsize=(3.5, 1.16), dpi=300)
        axes = pt.add_axes_as_grid(self._fig, 2, 6, m_xc=0.01, m_yc=0.01)

        self._imshows = []
        for i, ax in enumerate( list(axes.flatten()) ):
            # disable legends
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            if i >= n_outputs:
                self._imshows.append(ax.imshow(np.zeros((24, 24))))
            else:
                self._imshows.append(ax.imshow(ut.sigmoid(net._V[i].reshape((24, 24)))))
        plt.show(block=False)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def update(self, weights):
        weights = weights.reshape((-1, 24, 24))
        for i, imshow in enumerate(self._imshows):
            if i <= len(weights):
                imshow.set_data(weights[i])

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

class WeightPCAPlotter():
    def __init__(self, X, n_outputs):
        # set up figure for PCA
        self._fig, self._ax = plt.subplots(1)
        colors = ["C0", "C1", "C2", "C3"]
        Y_color = np.empty(Y.shape, dtype="object")
        for i, label in enumerate(labels):
            Y_color[Y == label] = colors[i%len(colors)]
        self._pca = PCA(n_components=2)
        self._pca.fit(X)
        X_pca = self._pca.transform(X)

        self._ax.set_xlabel("PC 1")
        self._ax.set_ylabel("PC 2")

        self._scatter_constant = self._ax.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_color,alpha=0.5, marker="o", s=2)
        self._scatter_variable = self._ax.scatter(np.zeros(n_outputs), np.zeros(n_outputs), c="black", marker="o", s=4)
        import matplotlib.patches as mpatches
        self._fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), fancybox=True, ncol=1+len(labels), handles=[mpatches.Patch(color=Y_color[i], label=labels[i]) for i in range(len(labels))] + [mpatches.Patch(color="black", label="Weights")])
        plt.show(block=False)

    def update(self, weights):
        weights_pca = self._pca.transform(weights)
        self._scatter_variable.set_offsets(weights_pca)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
    

# train
pca_plotter = WeightPCAPlotter(X, n_outputs)
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
    

