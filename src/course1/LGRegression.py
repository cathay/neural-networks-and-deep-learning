
import numpy as np

class LGRegression:
    def __init__(self):
        print "LG"

    def train(self, X, Y, W, b):
        m = num_of_samples = X.shape[0]
        Z = np.dot(W.T, X) + b
        A = np.vectorize(self.sigmoid)(Z)
        dZ = A - Y
        dW = 1/m * np.dot(X, dZ.T)
        db = 1/m * np.sum(dZ)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))