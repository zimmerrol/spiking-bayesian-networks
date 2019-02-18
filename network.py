import numpy as np
import utility as ut

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

        self._b += self._delta_t * self._eta_b * (self._r_net * self._m_k - ut.dirac(z - 1))
        self._V += self._delta_t * self._eta_v * ut.dirac(z - 1) * (inputs.T - ut.sigmoid(self._V))

        return z
