from tensorflow.keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm

# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dirac(x):
    return np.isclose(x, 0).astype(np.float32)


def generate_spike_trains(X_freq, T, T_image = 0.250, delta_T = 0.0001):
    """
    Parameters
        ----------
        X_freq : ~numpy.ndarray
            Image sequence

        T : float
            total time length of the spiek train seconds

        T_image : float
            duration to show each image in seconds

        delta_T : float
            sampling freq discretization of the poisson process
    """
    n_samples = int(np.ceil(T / T_image))
    sample_steps = int(np.ceil(T_image/delta_T))

    n_steps = n_samples * sample_steps

    # generate time dependant image showing rates
    rates = np.zeros((n_steps, X_freq.shape[1]))
    sample_indices = np.random.randint(0, len(X_freq), size=n_samples)
    for i, sample_index in enumerate(sample_indices):
        rates[i*sample_steps:(i+1)*sample_steps] = X_freq[sample_index]

    # now generate the spike trains
    p = np.random.uniform(0.0, 1.0, size=rates.shape)
    y = (rates*delta_T > p).astype(np.float32)

    return y

# ------------------------------------------------------------------ #
# plot
# ------------------------------------------------------------------ #

def plot_spiketrain(delta_T, spiketrain_nd, tmin = 0.0, tmax = None):
    """
    Parameters
        ----------
        delta_T : float
            sampling freq discretization of spike train

        spiketrain_nd : numpy.ndarray
            first dim spike index, second dim is neuron id

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

# ------------------------------------------------------------------ #
# network
# ------------------------------------------------------------------ #

class Network:
    def __init__(self, n_inputs, n_outputs, delta_t, r_net, m_k, eta_v, eta_b):
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._delta_t = delta_t
        self._r_net = r_net
        self._m_k = m_k
        self._eta_v = eta_v
        self._eta_b = eta_b

        self._V = np.random.normal(scale=1e-3, size=(self._n_outputs, self._n_inputs))
        self._b = np.zeros((self._n_outputs, 1))

    def step(self, inputs):
        inputs = inputs.reshape((-1, 1))
        assert len(inputs) == self._n_inputs, "Input length does not match"

        u = np.dot(self._V, inputs) + self._b

        z = np.zeros((self._n_outputs, 1))

        p_z = np.exp(u) / np.sum(np.exp(u)) * self._delta_t * self._r_net

        sum_p_z = np.cumsum(p_z)
        diff = sum_p_z - np.random.uniform(0, 1, 1) > 0

        k = np.argmax(diff)

        if diff[k]:
            z[k] = 1.0

        self._b += self._delta_t * self._eta_b * (self._r_net * self._m_k - dirac(z - 1))
        self._V += self._delta_t * self._eta_v * dirac(z - 1) * (inputs.T - sigmoid(self._V))

        return z
