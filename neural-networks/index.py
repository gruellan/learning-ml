from neural_net import *


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
