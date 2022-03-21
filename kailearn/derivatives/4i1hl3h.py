# 4 Input Features, 1 Hidden Layer, 3 Neurons per hidden layer
import numpy as np
# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

print(dvalues)
# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T
                    
# sum weights of given input
# and multiply by the passed in gradient for this neuron
dx0 = sum(weights[0])*dvalues[0]
dx1 = sum(weights[1])*dvalues[0]
dx2 = sum(weights[2])*dvalues[0]
dx3 = sum(weights[3])*dvalues[0]
dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)

dinputs = np.dot(dvalues, weights.T)
print(dinputs)

inputs = np.array([[1, 2, 3, 2.5],
[2., 5., -1., 2],
[-1.5, 2.7, 3.3, -0.8]])

dweights = np.dot(inputs.T, dvalues)
print(dweights)

# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(dbiases)

# Example layer output
z = np.array([[1, 2, -3, -4],
[2, -7, -1, 3],
[-1, 2, 5, -1]])


# ReLU activation's derivative
drelu = np.zeros_like(z)
drelu[z > 0] = 1.0
print(drelu)
# The chain rule
drelu *= dvalues
print(drelu)