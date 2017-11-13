import mnist_loader
import network
import matrix_network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#net = network.Network([784, 30, 10])
#net = network.Network([784, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

nets = matrix_network.MatrixNetwork([2, 3, 1])
nets.feedforward()

