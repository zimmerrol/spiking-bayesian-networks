import numpy as np
from matplotlib import pyplot as plt
import utility as ut
import network as nt

# sample spike train plot
test = np.ones(shape=(100))*0.0
test = np.vstack((test, np.ones(shape=(100))*1.0) )
test = test.T
X_spikes = ut.generate_spike_trains(test, 1000, delta_T=1e-2)

fig, ax = ut.plot_spiketrain(X_spikes, 1e-2, tmax = 1)
plt.show(block=True)

# mnist
labels = [2, 4, 8]
(x_train, y_train), (x_test, y_test) = ut.mnist.load_data()
selection = np.any([y_test == label for label in labels], axis=0)
X = x_test[selection]
Y = y_test[selection]
X = X.reshape((len(X), -1)) / 255.0
X = (X > 0.5).astype(np.float32)

X_spikes = generate_spike_trains(X, 1000, delta_T=1e-2)


net = nt.Network(28*28, 10, 1e-2, 10, 1/10, eta_v=1e-1, eta_b=1e-1*10)

for i in ut.tqdm(range(len(X_spikes))):
    net.step(X_spikes[i])


net._V


weights = net._V.reshape((-1, 28, 28))



for i in range(10):
    plt.imshow(sigmoid(weights[i]))

plt.show(block=True)


sigmoid(net._V+5)

z = np.zeros((10, 1))
z[0] = 1



np.isclose(0, 1e-190, )

