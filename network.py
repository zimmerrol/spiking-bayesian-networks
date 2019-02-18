import numpy as np
import utility as ut
from abc import ABC, abstractclassmethod

# ------------------------------------------------------------------ #
# network
# ------------------------------------------------------------------ #

class BinaryWTANetwork():
    """
    Implementation of the (soft) Winner Takes All (WTA) network, assuming Bernoulli distributions
    """

    def __init__(self, n_inputs, n_outputs, delta_t, r_net, m_k, eta_v, eta_b):
        """
            Parameters
            ----------
            delta_t : float
                numeric integration time step size

            r_net : float
                global target firing rate of the whole network

            m_k :  numpy.ndarray
                size n_outputs
                has to sum up to one, propto relative target firing rate of
                neuron k

            eta_v : float
                learning rate of v eq (8)

            eta_b : float
                learing rate of b eq. (7)
        """

        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._delta_t = delta_t
        self._r_net = r_net
        self._m_k = m_k
        self._eta_v = eta_v
        self._eta_b = eta_b

        if np.isscalar(self._m_k):
            self._m_k = np.ones((n_outputs, 1)) * self._m_k

        assert len(self._m_k) == n_outputs, "Length of m_ks does not match number of output neurons"

        self._V = np.random.normal(scale=1e-3, size=(self._n_outputs, self._n_inputs))
        self._b = np.zeros((self._n_outputs, 1))

    def step(self, inputs):
        inputs = inputs.reshape((-1, 1))
        assert len(inputs) == self._n_inputs, "Input length does not match"

        # u = V * input + b
        u = np.dot(self._V, inputs) + self._b

        z = np.zeros((self._n_outputs, 1))

        # p \propto softmax(u); eq. (4)
        p_z = np.exp(u) / np.sum(np.exp(u) + 1e-8) * self._delta_t * self._r_net

        # sample from softmax distribution, i.e. choose a single neuron to spike
        sum_p_z = np.cumsum(p_z)
        diff = sum_p_z - np.random.uniform(0, 1, 1) > 0
        k = np.argmax(diff)

        if diff[k]:
            z[k] = 1.0

        self._b += self._delta_t * self._eta_b * (self._r_net * self._m_k - ut.dirac(z - 1))
        self._V += self._delta_t * self._eta_v * ut.dirac(z - 1) * (inputs.T - ut.sigmoid(self._V))

        return z


class ContinuousWTANetwork():
    """
    Implementation of the (soft) Winner Takes All (WTA) network, assuming Gaussian distributions
    """

    def __init__(self, n_inputs, n_outputs, delta_t, r_net, m_k, eta_v, eta_b, eta_beta):
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._delta_t = delta_t
        self._r_net = r_net
        self._m_k = m_k
        self._eta_v = eta_v
        self._eta_b = eta_b
        self._eta_beta = eta_beta

        if np.isscalar(self._m_k):
            self._m_k = np.ones((n_outputs, 1)) * self._m_k

        assert len(self._m_k) == n_outputs, "Length of m_ks does not match number of output neurons"

        self._V = np.random.normal(scale=1e-3, size=(self._n_outputs, self._n_inputs))
        self._b = np.random.normal(scale=1e-3, size=(self._n_outputs, 1))
        self._beta = np.ones((self._n_inputs, 1))

    def step(self, inputs):
        inputs = inputs.reshape((-1, 1))
        assert len(inputs) == self._n_inputs, "Input length does not match"

        # u = V * input + b
        u = 0.5 * np.dot(self._V, self._beta * inputs) + self._b

        z = np.zeros((self._n_outputs, 1))

        # p = softmax(u)
        p_z = np.exp(u) / np.sum(np.exp(u) + 1e-8) * self._delta_t * self._r_net

        # sample from softmax distribution
        sum_p_z = np.cumsum(p_z)
        diff = sum_p_z - np.random.uniform(0, 1, 1) > 0
        k = np.argmax(diff)

        if diff[k]:
            z[k] = 1.0

        self._b += self._delta_t * self._eta_b * (self._r_net * self._m_k - ut.dirac(z - 1))
        self._V += self._delta_t * self._eta_v * ut.dirac(z - 1) * self._beta.T * (inputs.T - self._V)
        self._beta += self._delta_t * self._eta_beta * (np.dot(self._V.T**2, z) + inputs * np.dot(self._V.T, z) - 0.5*inputs**2 + 1.0 / self._beta)

        return z
