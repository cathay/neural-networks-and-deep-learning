import random

# Third-party libraries
import numpy as np

class MatrixNetwork(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x):
        return self.activations(x)[-1][-1]

    def activations(self, x):
        self.debug()
        activations = [x]
        a = x
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            zs.append(z)
            activations.append(a)
        print activations
        return zs, activations

    def backprop(self, x, y):
        zs, activations = self.activations(x)
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])

    def debug(self):
        print np.matrix(self.weights)
        print("......")
        print np.matrix(self.biases)
        print(".....")

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(w):
    e = np.exp(w)
    return e / np.sum(e)

print softmax(np.array([1, -2, 0.5]))


#nets = MatrixNetwork([2, 3, 1])
#input1 = np.array([[1,1]]).transpose()
#print input1
#nets.feedforward(input1)
a = np.random.randn(2, 3) # a.shape = (2, 3)
b = np.random.randn(4, 1) # b.shape = (2, 1)
#print a
#print a.reshape(a.shape[0], -1).T
#print b
#c = a + b
#print c

c = np.array([
        [0,0],
        [1,0],
        [2.61,-1.28],
        [-0.59,2.1]
    ])

print c.shape[0]
X_expanded = np.zeros((c.shape[0], 6))

print X_expanded.shape[1]
print c[2][1]
print c[2,1]
for i in range(c.shape[0]):
    X_expanded[i,0] = c[i][0]
    X_expanded[i,1] = c[i][1]
    X_expanded[i,2] = X_expanded[i,0] * X_expanded[i,0]
    X_expanded[i,3] = X_expanded[i,1] * X_expanded[i,1]
    X_expanded[i,4] = X_expanded[i,0] * X_expanded[i,1]
    X_expanded[i,5] = 1

print X_expanded
