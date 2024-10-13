Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:

Keras pakages a number of deep leanring models alongside pre-trained weights into an applications module.

You must use one of the applications listed in Keras Applications
Your script must save your trained model in the current working directory as cifar10.h5
Your saved model should be compiled
Your saved model should have a validation accuracy of 87% or higher
Your script should not run when the file is imported

X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data, where m is the number of data points
Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
Returns: X_p, Y_p
X_p is a numpy.ndarray containing the preprocessed X
Y_p is a numpy.ndarray containing the preprocessed Y
