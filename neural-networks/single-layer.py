import numpy as np

'''
Single neuron layer with one data point
'''

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Dot product
# y = mx + b
output = np.dot(weights, inputs) + biases
print(output)

'''
Single neuron layer with batch of data points
'''
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

# Matrix product takes all of the combos of rows from the left matrix and
# columns from the right matrix, performing the dot product on them.
# Requires num cols of first matrix == num rows of second matrix
# but currently ours have the same shape (3, 4) so we transpose weights.
# Now we have inputs = (3, 4) and weights = (4, 3) so we can perform matrix product
layer_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer_outputs)
