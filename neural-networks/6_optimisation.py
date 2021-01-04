from nnfs.datasets import vertical_data, spiral_data
from neural_net import *

# Init
X, y = spiral_data(samples=100, classes=3)
relu = ReLU()
softmax = Softmax()
loss_func = CategoricalCrossEntropyLoss()
loss = accuracy = 0
optimiser = SGD()

# Create two layers
layer = DenseLayer(2, 3)
layer_2 = DenseLayer(3, 3)

# Helpers
lowest_loss = float("Inf")
best_layer_weights, best_layer_biases = layer.weights.copy(), layer.biases.copy()
best_layer_2_weights, best_layer_2_biases = layer_2.weights.copy(), layer_2.biases.copy()

for iter in range(10000):
    # Update weights
    layer.weights += 0.05 * np.random.randn(2, 3)
    layer.biases += 0.05 * np.random.randn(1, 3)
    layer_2.weights += 0.05 * np.random.randn(3, 3)
    layer_2.biases += 0.05 * np.random.randn(1, 3)

    # Forward pass
    layer.forward(X)
    relu.forward(layer.output)

    layer_2.forward(relu.output)
    softmax.forward(layer_2.output)

    # Calculate loss and accuracy
    loss = loss_func.calc(softmax.output, y)
    predictions = np.argmax(softmax.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Save new weights and biases if they reduce loss
    if loss < lowest_loss:
        lowest_loss = loss
        best_layer_weights, best_layer_biases = layer.weights.copy(), layer.biases.copy()
        best_layer_2_weights, best_layer_2_biases = layer_2.weights.copy(), layer_2.biases.copy()
    else:
        layer.weights, layer.biases = best_layer_weights.copy(), best_layer_biases.copy()
        layer_2.weights, layer_2.biases = best_layer_2_weights.copy(), best_layer_2_biases.copy()

    print("Loss:", loss)
    print('Accuracy:', accuracy)
print("Loss:", loss)
print('Accuracy:', accuracy)
