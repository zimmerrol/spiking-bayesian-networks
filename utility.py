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
    sample_indices = np.random.randint(0, len(X_freq), size=n_samples)
    rates = np.repeat(X_freq[sample_indices], sample_steps, axis=0)
    
    # now generate the spike trains
    p = np.random.uniform(0.0, 1.0, size=rates.shape)
    y = (rates*delta_T > p).astype(np.float32)

    return y

# ------------------------------------------------------------------ #
# plot
# ------------------------------------------------------------------ #

def plot_spiketrain(delta_T, spiketrain_nd, tmin = 0, tmax = None):
    """
    Parameters
        ----------
        delta_T : float
            sampling freq discretization of spike train

        spiketrain_nd : numpy.ndarray
            first dim is neuron id, second is spike

        tmin : int
            plotrange in ms

        tmax : int
            plotrange in ms
    """

    fig, ax = plt.subplots()

    t_ms_total = int(spiketrain_nd.shape[1]*delta_T)

    # neurons_to_plot = range(28*28)[0:-1:20]
    neurons_to_plot = np.arange(spiketrain_nd.shape[0])
    for i in neurons_to_plot:
        ax.scatter(np.arange(a)[X_spikes[:a, i] == 1.0],
            i*np.ones(np.sum(X_spikes[:a, i] == 1.0)),
            s=0.5, c='C1')

        ax.set_xlabel('ms')
        ax.set_ylabel(r'Neuron $i$')

    return fig, ax