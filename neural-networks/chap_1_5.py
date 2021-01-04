from neural_net import *
from nnfs.datasets import spiral_data

# Init
X, y = spiral_data(samples=100, classes=3)
relu = ReLU()
softmax = Softmax()
loss = CategoricalCrossEntropyLoss()

# Create two dense layers, layer_2 = (3, 3) because the input == output of first layer
layer = DenseLayer(2, 3)
layer_2 = DenseLayer(3, 3)

# Forward pass through dense layer, then through relu
layer.forward(X)
relu.forward(layer.output)

# Forward pass through layer_2, then through softmax
layer_2.forward(relu.output)
softmax.forward(layer_2.output)

# Calculate loss on the final output
loss_value = loss.calc(softmax.output, y)

# Calculate accuracy
predictions = np.argmax(softmax.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print(relu.output[:5])
print("Loss:", loss_value)
print('Accuracy:', accuracy)
