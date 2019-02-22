"""
MIT License

Copyright (c) 2019 Roland Zimmermann, Laurenz Hemmen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from tensorflow.keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dirac(x):
    return np.isclose(x, 0).astype(np.float32)


def generate_spike_trains(frequencies, T, T_image=0.250, delta_T=1e-2, batch_size=10):
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

    # number of samples to show in time T
    n_samples = int(np.ceil(T / T_image))
    # steps per sample
    sample_steps = int(np.ceil(T_image/delta_T))
   
    # generate time dependant image showing rates
    sample_indices = np.random.randint(0, len(frequencies), size=n_samples)
    frequencies = frequencies[sample_indices]

    n_batches = int(n_samples / batch_size)
    for i in range(n_batches):
        frequencies_index_start = i * batch_size
        frequencies_index_end = min((i+1) * batch_size, n_samples)
        rates = np.repeat(frequencies[frequencies_index_start:frequencies_index_end], sample_steps, axis=0)

        # now generate the spike trains
        p = np.random.uniform(0.0, 1.0, size=rates.shape)
        y = (rates*delta_T > p).astype(np.float32)
        
        yield y


def generate_constant_trains(frequencies, T, T_image=0.250, delta_T=1e-2, batch_size=10):
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

    # number of samples to show in time T
    n_samples = int(np.ceil(T / T_image))
    # steps per sample
    sample_steps = int(np.ceil(T_image/delta_T))
   
    # generate time dependant image showing rates
    sample_indices = np.random.randint(0, len(frequencies), size=n_samples)
    frequencies = frequencies[sample_indices]

    n_batches = int(n_samples / batch_size)
    for i in range(n_batches):
        frequencies_index_start = i * batch_size
        frequencies_index_end = min((i+1) * batch_size, n_samples)
        rates = np.repeat(frequencies[frequencies_index_start:frequencies_index_end], sample_steps, axis=0)

        yield rates


def generate_bars(num_samples, height, width, p):
    X = np.zeros((num_samples, height, width))
    bars_x = np.random.uniform(size=(num_samples, width, 1)) < p
    bars_y = np.random.uniform(size=(num_samples, height, 1)) < p
    print("\% horizontal bars: {}".format(np.sum(bars_x)/num_samples ) )
    print("\% vertical bars: {}".format(np.sum(bars_y)/num_samples ) )
    
    X[bars_x[:, :, 0], :] = 1.0
    for i, x in enumerate(X):
        x[:, bars_y[i,:, 0]] = 1.0
    return X


def load_mnist(h=28, w=28, labels=None, train=False, frequencies=False):
    """
    Loads images from MNIST dataset.
    :param h: height (default=28)
    :param w: width (default=28)
    :param labels: list of labels to load. None equals to all labels. (default=None)
    :param train: Use training or validaition data. (default=False)
    :param frequencies: Return poisson frequencies. (default=False)
    :return: Input data, labels
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if labels is None:
        labels = list(range(10))

    selection = [y_test == label for label in labels]

    minimum_length = min(np.sum(selection, axis=1))
    selection = np.any([np.all((item, np.cumsum(item) < minimum_length), axis=0) for item in selection], axis=0)

    if train:
        x = x_train[selection]
        y = y_train[selection]
    else:
        x = x_test[selection]
        y = y_test[selection]

    # cut off margins
    if h < 28 and w < 28:
        x = x[:, (28 - h) // 2:-(28 - h) // 2, (28 - w) // 2:-(28 - w) // 2]

    x = x / 255.0
    x = (x > 0.5).astype(np.float32)

    if frequencies:
        x_frequencies = x * 70.0 + 20.0
        return x_frequencies, y
    else:
        return x, y


# test code. creates two trains, one with frequency 10 and one with 100
if __name__ == '__main__':
    # sample spike train plot
    test = np.ones(shape=(100))*10.0
    test = np.vstack((test, np.ones(shape=(100))*100.0)).T
    X_spikes = generate_spike_trains(test, 1000, delta_T=1e-2)
    from plot import plot_spiketrain
    fig, ax = plot_spiketrain(X_spikes, 1e-2, tmax = 1)
    plt.show(block=True)

