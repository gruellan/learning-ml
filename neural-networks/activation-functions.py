import numpy as np
from scipy.special import softmax


# Used for regression
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Used for multi-class classification
# Converts a vector of non-normalised numbers into a vector of normalised probabilities
class Softmax:
    # Softmax of a vector = exponent(vector) / exponent(vector).sum()
    def forward(self, inputs):
        # Get probabilities
        exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalise them
        self.output = exponents / np.sum(exponents, axis=1, keepdims=True)


class DenseLayer:
    def __init__(self, n_inputs, neurons):
        # Init weights as (inputs, neurons) instead of traditional (neurons, inputs)
        # so we don't have to transpose everytime we do a forward pass
        self.weights = 0.01 * np.random.randn(n_inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        # Calculate outputs
        self.output = np.dot(inputs, self.weights) + self.biases


# From: https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py
def spiral(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


# Init
X, y = spiral(samples=100, classes=3)
relu = ReLU()
softmax = Softmax()

# Create two dense layers, layer_2 = (3, 3) because the input == output of first layer
layer = DenseLayer(2, 3)
layer_2 = DenseLayer(3, 3)

# Forward pass through dense layer, then through relu
layer.forward(X)
relu.forward(layer.output)

# Forward pass through layer_2, then through softmax
layer_2.forward(relu.output)
softmax.forward(layer_2.output)

print(relu.output[:5])
