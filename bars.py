import numpy as np 
import matplotlib.pyplot as plt
import time

num_samples = 100
dim = 8
p =1.4/dim
X = np.zeros((num_samples,dim,dim))
bars = np.random.uniform( size=(num_samples, dim, 2) ) < p
print(bars.shape, X.shape)
X[bars[:, :, 0], :] = 1.0
for i,x in enumerate(X):
    x[:,bars[i, :, 1]] = 1.0

fig, ax = plt.subplots()
for i in range(num_samples):
    print(bars[i])
    ax.imshow(X[i])
    plt.pause(0.5)
plt.show()

