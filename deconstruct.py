import numpy as np

x = np.load("moving_mnist.npy")

# (20,10000,64,64)
print(x.shape)

np.set_printoptions(threshold=np.inf)
print(x[0])
