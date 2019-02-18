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
    print(f'Generating spike trains for {n_samples} images')

    n_steps = n_samples * sample_steps

    # generate time dependant image showing rates
    sample_indices = np.random.randint(0, len(seq), size=n_samples)
    rates = np.repeat(seq[sample_indices], sample_steps, axis=0)

    # now generate the spike trains
    p = np.random.uniform(0.0, 1.0, size=rates.shape)
    y = (rates*delta_T > p).astype(np.float32)

    return y
