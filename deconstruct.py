import numpy as np

# import keras

# keras.utils.get_file(
#     "moving_mnist.npy",
#     "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
# )
x = np.load("moving_mnist.npy")

# (20,10000,64,64)
print(x.shape)

np.set_printoptions(threshold=np.inf)
print(x[0][0])
