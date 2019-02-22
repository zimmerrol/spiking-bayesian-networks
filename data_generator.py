import numpy as np


class DataGenerator:
    """
    Handles the generation of data to feed it into a (spiking) bayesian network with spiking output.
    """

    def __init__(self, data, length, delta_t, t_image=0.250, spiking=False):
        self._data = data
        self._indices = np.random.uniform(0, len(data), size=length).astype(np.int32)
        self._delta_t = delta_t
        self._t_image = t_image
        self._spiking = spiking
        self._cache = SpikeTrainCache(delta_t)

    def __getitem__(self, time):
        if time in self._cache:
            return self._cache[time]

        image_index = self._indices[int(time / self._t_image)]

        if self._spiking:
            rate = self._data[image_index]
            y = (np.random.uniform(0, 1, self._data.shape[1]) < self._delta_t * rate).astype(np.float32)
        else:
            y = self._data[image_index]

        self._cache[time] = y

        return y


class SpikeTrainCache:
    """
    Caches spike trains in order to implement learning rules that need to access past inputs.
    """

    def __init__(self, delta_t):
        self._delta_t = delta_t
        self._trace = {}

    def _bin_time(self, time):
        return int(time / self._delta_t)

    def __contains__(self, time):
        return self._bin_time(time) in self._trace

    def __getitem__(self, time):
        return self._trace[self._bin_time(time)]

    def __setitem__(self, time, value):
        self._trace[self._bin_time(time)] = value

