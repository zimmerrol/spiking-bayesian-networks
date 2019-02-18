
# coding: utf-8

# In[6]:


from tensorflow.keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm


# In[4]:


labels = [2, 4, 8]


# In[7]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
selection = np.any([y_test == label for label in labels], axis=0)
X = x_test[selection]
Y = y_test[selection]

X = X.reshape((len(X), -1)) / 255.0
X = (X > 0.5).astype(np.float32)

X_freq = 20 + X*70


# In[20]:


def generate_spike_trains(X_freq, T, T_image = 0.250, delta_T = 0.01):
    n_samples = int(np.ceil(T / T_image))
    sample_steps = int(np.ceil(T_image/delta_T))
    
    n_steps = n_samples * sample_steps
    
    # generate rates
    rates = np.zeros((n_steps, X_freq.shape[1]))
    sample_indices = np.random.randint(0, len(X_freq), size=n_samples)
    rates = np.stack(X_freq[sample_indices])
    
    # now generate the spike trains
    p = np.random.uniform(0.0, 1.0, size=rates.shape)
    y = (rates*delta_T > p).astype(np.float32)
        
    return y


# In[21]:


X_spikes = generate_spike_trains(X_freq, 1000, delta_T=1e-2)


# In[19]:


a = 50
for i in range(28*28):
    plt.scatter(np.arange(a)[X_spikes[:a, i] == 1.0], i*np.ones(np.sum(X_spikes[:a, i] == 1.0)), s=0.5, c='C1')
plt.show()


# In[236]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dirac(x):
    return np.isclose(x, 0).astype(np.float32)


# In[237]:


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


# In[238]:


net = Network(28*28, 10, 1e-2, 10, 1/10, eta_v=1e-1, eta_b=1e-1*10)


# In[239]:


for i in tqdm(range(len(X_spikes))):
    net.step(X_spikes[i])


# In[240]:


net._V


# In[241]:


weights = net._V.reshape((-1, 28, 28))


# In[242]:


for i in range(10):
    plt.imshow(sigmoid(weights[i]))
    plt.show()


# In[179]:


sigmoid(net._V+5)


# In[113]:


z = np.zeros((10, 1))
z[0] = 1


# In[189]:


np.isclose(0, 1e-190, )

