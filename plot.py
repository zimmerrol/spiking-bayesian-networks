import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import utility as ut
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ #
# matplotlib helpers
# ------------------------------------------------------------------ #

# cm to fraction (0-1)
def cmtf(cm, figsize_inch = 1):
    return cm/2.54/figsize_inch

# margins in cm
def add_axes_as_grid(fig, rows, cols, m_xl=0, m_yl=0, m_xr=0, m_yr=0, m_xc=0, m_yc=0):

    # margins, left, center, right
    m_xl = cmtf(m_xl, fig.get_figwidth())
    m_xc = cmtf(m_xc, fig.get_figwidth())
    m_xr = cmtf(m_xr, fig.get_figwidth())
    m_yl = cmtf(m_yl, fig.get_figheight())
    m_yc = cmtf(m_yc, fig.get_figheight())
    m_yr = cmtf(m_yr, fig.get_figheight())

    h = (1-m_yl-(m_yc*(rows-1))-m_yr)/rows
    w = (1-m_xl-(m_xc*(cols-1))-m_xr)/cols
    # left bot w h
    rect = [m_xl, m_yl, w, h]
    axes = []
    for i in np.arange(0, rows):
        axes.append([])
        rect[0] = m_xl
        rect[1] = m_yl+i*(h+m_yc)
        for j in np.arange(0, cols):
            axes[i].append(fig.add_axes(rect))
            rect[0] += w + m_xc

    return np.array(axes)

# ------------------------------------------------------------------ #
# plot
# ------------------------------------------------------------------ #


def plot_spiketrain(spiketrain_nd, delta_T, tmin = 0.0, tmax = None):
    """
        Parameters
        ----------
        spiketrain_nd : numpy.ndarray
            first dim spike index, second dim is neuron id

        delta_T : float
            sampling freq discretization of spike train

        tmin : float
            plotrange in s

        tmax : float
            plotrange in s
    """

    fig, ax = plt.subplots()

    t_total = np.ceil(spiketrain_nd.shape[0]*delta_T)

    if tmax is None:
        tmax = t_total

    x_values = np.arange(tmin, tmax, delta_T)
    # neurons_to_plot = range(28*28)[0:-1:20]
    neurons_to_plot = np.arange(spiketrain_nd.shape[1])
    for i in neurons_to_plot:
        selection = spiketrain_nd[int(tmin/delta_T):int(tmax/delta_T), i] == 1.0
        ax.scatter(x_values[selection],
            i*np.ones(np.sum(selection)),
            s=0.5, c='C1')

        ax.set_xlabel(r'Time $[s]$')
        ax.set_ylabel(r'Neuron $i$')

    return fig, ax


class WeightPlotter():
    def __init__(self, weights):
        self._fig = plt.figure(figsize=(3.5, 1.16), dpi=300)
        axes = add_axes_as_grid(self._fig, 2, 6, m_xc=0.01, m_yc=0.01)

        self._weight_shape = weights.shape[1:]

        self._imshows = []
        for i, ax in enumerate( list(axes.flatten()) ):
            # disable legends
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            if i >= len(weights):
                self._imshows.append(ax.imshow(np.zeros(self._weight_shape), vmin=0, vmax=1))
            else:
                self._imshows.append(ax.imshow(ut.sigmoid(weights[i].reshape(self._weight_shape)), vmin=0, vmax=1))
        plt.show(block=False)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def update(self, weights):
        weights = weights.reshape((-1, *self._weight_shape))
        for i, imshow in enumerate(self._imshows):
            if i <= len(weights):
                imshow.set_data(weights[i])

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


class WeightPCAPlotter():
    def __init__(self, X, Y, n_outputs, labels, annotations=False):
        # set up figure for PCA
        self._fig, self._ax = plt.subplots(1)
        colors = ["C0", "C1", "C2", "C3"]
        Y_color = np.empty(Y.shape, dtype="object")
        for i, label in enumerate(labels):
            Y_color[Y == label] = colors[i%len(colors)]
        self._pca = PCA(n_components=2)
        self._pca.fit(X)
        X_pca = self._pca.transform(X)

        if annotations:
            from matplotlib import offsetbox
            shown_images = np.array([[1., 1.]])
            dim = int(np.sqrt(X[i].shape))
            for i in range(X.shape[0]):
                dist = np.sum((X_pca[i] - shown_images) ** 2, 1)
                if np.min(dist) < 5e-1:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X_pca[i]] ]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(X[i].reshape((dim, dim)), zoom=1.5),
                    X_pca[i])
                self._ax.add_artist(imagebox)

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


class CurvePlotter():
    def __init__(self, x=None, y=None, x_label=None, y_label=None):
        # set up figure for PCA
        self._fig, self._ax = plt.subplots(1)

        self._line = Line2D([], [], color='C0')
        self._ax.add_line(self._line)
        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel(y_label)

        plt.show(block=False)

    def update(self, x, y=None):
        if y is None:
            y = x
            x = np.arange(len(y))

        y = np.array(y)
        x = np.array(x)

        self._ax.set_xlim([x.min(), x.max()])
        self._ax.set_ylim([y.min(), y.max()])

        self._line.set_data(x, y)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()