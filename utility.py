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


def generate_spike_trains(seq, T, T_image = 0.250, delta_T = 0.0001,
    f_0=20.0, f_1=90.0):
    """
    Parameters
        ----------
        seq : ~numpy.ndarray
            Image sequence, pixel between 0-1

        T : float
            total time length of the spiek train seconds

        T_image : float
            duration to show each image in seconds

        delta_T : float
            sampling freq discretization of the poisson process

        f_0  : float
            firing frequency if pixel is 0

        f_1  : float
            firing frequency if pixel is 1


    """
    seq = f_0 + seq*(f_1-f_0)

    n_samples = int(np.ceil(T / T_image))
    sample_steps = int(np.ceil(T_image/delta_T))

    n_steps = n_samples * sample_steps

    # generate time dependant image showing rates
    sample_indices = np.random.randint(0, len(seq), size=n_samples)
    rates = np.repeat(seq[sample_indices], sample_steps, axis=0)

    # now generate the spike trains
    p = np.random.uniform(0.0, 1.0, size=rates.shape)
    y = (rates*delta_T > p).astype(np.float32)

    return y

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
