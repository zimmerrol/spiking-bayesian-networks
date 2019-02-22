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

