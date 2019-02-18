from tensorflow.keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt

# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dirac(x):
    return np.isclose(x, 0).astype(np.float32)


def generate_spike_trains(frequencies, T, T_image = 0.250, delta_T = 0.0001):
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


    """

    n_samples = int(np.ceil(T / T_image))
    sample_steps = int(np.ceil(T_image/delta_T))
    print(f'Generating spike trains for {n_samples} images')

    n_steps = n_samples * sample_steps

    # generate time dependant image showing rates
    sample_indices = np.random.randint(0, len(frequencies), size=n_samples)
    rates = np.repeat(frequencies[sample_indices], sample_steps, axis=0)

    # now generate the spike trains
    p = np.random.uniform(0.0, 1.0, size=rates.shape)
    y = (rates*delta_T > p).astype(np.float32)

    return y


def generate_bars(num_samples, height, width, p=1.0):
    X = np.zeros((num_samples, height, width))
    bars_y = np.random.uniform(size=(num_samples, height)) < p
    bars_x = np.random.uniform(size=(num_samples, width)) < p

    X[bars_x[:, :, 0], :] = 1.0
    for i, x in enumerate(X):
        x[:, bars_y[i, :, 1]] = 1.0

    return X


# test code. creates two trains, one with frequency 10 and one with 100
if __name__ == '__main__':
    # sample spike train plot
    test = np.ones(shape=(100))*10.0
    test = np.vstack((test, np.ones(shape=(100))*100.0)).T
    X_spikes = generate_spike_trains(test, 1000, delta_T=1e-2)
    from plot import plot_spiketrain
    fig, ax = plot_spiketrain(X_spikes, 1e-2, tmax = 1)
    plt.show(block=True)

