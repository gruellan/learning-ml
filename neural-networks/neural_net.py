import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, neurons):
        # Init weights as (inputs, neurons) instead of traditional (neurons, inputs)
        # so we don't have to transpose everytime we do a forward pass
        self.weights = 0.01 * np.random.randn(n_inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        # Save inputs for backpropagation
        self.inputs = inputs

        # z = xw0 + xw1 + xw2 + b
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Param gradients
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Value gradients
        self.dinputs = np.sum(dvalues, self.weights.T)


class SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += - self.learning_rate * layer.dbiases


# Activation function used for regression
class ReLU:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Make a copy before we modify the originals
        self.dinputs = dvalues.copy()

        # Add zeros where input value is negative
        self.dinputs[self.inputs <= 0] = 0


# Activation function used for multi-class classification
# Converts a vector of non-normalised numbers into a vector of normalised probabilities
class Softmax:
    def __init__(self):
        pass

    # Softmax of a vector = exponent(vector) / exponent(vector).sum()
    def forward(self, inputs):
        # Get probabilities
        exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalise them
        self.output = exponents / np.sum(exponents, axis=1, keepdims=True)


# Mean squared error - for regression

# Common loss class
class Loss:
    def calc(self, output, y):
        sample_losses = self.forward(output, y)

        # Return mean loss
        return np.mean(sample_losses)


# Categorical cross-entropy is used with softmax activation
# for multi-class classification
class CategoricalCrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # Clip to prevent div zero
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_probabilities = y_pred[
                range(samples),
                y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_probabilities = np.sum(y_pred * y_true, axis=1)

        return -np.log(correct_probabilities)
