import math
import matplotlib.pyplot 
import numpy as np
from numpy.core.overrides import array_function_dispatch
import dataset as ds
np.random.seed(4)

LAYER_START = 3.51  ## 0.10 init value; 1.6: best

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = LAYER_START * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    # Backwards Pass (backprop)
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    
    def backward(self, dvalues):
        # because we need to modify the original variable we should make a copy.
        self.dinputs = dvalues.copy()

        # Zero gradient for negative values.
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = prob
    
    def backwards(self, dvalues):

        # create un-init array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for i, (single_output, single_devalues) in \
            enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calcualte Jacobian matric of the output
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)

                # Calculate sample-wise gradient
                # and add it ot the array of sample gradients.
            self.dinputs[i] = np.dot(jacobian_matrix, single_devalues)

    

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CatagoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelyhoods = -np.log(correct_confidences)
        return negative_log_likelyhoods

    def backwards(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        labels = len(dvalues[0])

        # If lables are sparse, turn them into a on-hot-vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate Gradient
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CatagoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)


    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Cost:
    def forward(self, y_pred, y_true):
        total = len(y_pred)
        x1 = np.array(y_pred).T
        x2 = np.array(y_true)
        result = np.subtract(x2, x1)
        result = np.square(result)
        return np.sum(result) / total


# X, y = ds.create_data(100, 3)

# dense1 = Layer_Dense(2,3)
# activation1 = Activation_ReLU()

# dense2 = Layer_Dense(3, 3)
# activation2 = Activation_Softmax()

# dense1.forward(X)
# activation1.forward(dense1.output)

# dense2.forward(activation1.output)
# activation2.forward(dense2.output)

# cost_function = Cost()
# print('Cost: ', cost_function.forward(activation2.output, y))

# loss_function = Loss_CatagoricalCrossEntropy()
# loss = loss_function.calculate(activation2.output, y)
# print('Loss: ', loss)


# lowest_loss = 99999
# best_dense1_weights = dense1.weights.copy()
# best_dense1_baises = dense2.biases.copy()
# best_dense2_weights = dense1.weights.copy()
# best_dense2_baises = dense2.biases.copy()

# Loss_over_itt = []
# costs = []
# highest_cost = -1

# for i in range(30000):
#     dense1.weights += 0.025 * np.random.randn(2,3)
#     dense1.biases += 0.025 * np.random.randn(1,3)
#     dense2.weights += 0.025 * np.random.randn(3,3)
#     dense2.biases += 0.025 * np.random.randn(1,3)

#     dense1.forward(X)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)
#     activation2.forward(dense2.output)

#     loss = loss_function.calculate(activation2.output, y)
#     # print(loss)

#     cost = cost_function.forward(activation2.output, y)
#     if(cost > highest_cost):
#         costs.append(cost)
#         highest_cost = cost
#     else:
#         costs.append(highest_cost)


#     predicitions = np.argmax(activation2.output, axis=1)
#     accuracy = np.mean(predicitions==y)

#     Loss_over_itt.append(lowest_loss)

#     if loss < lowest_loss:
#         # print("New set of weights found, itteration: ", i," loss:", loss, " acc: ", accuracy)
#         best_dense1_weights = dense1.weights.copy()
#         best_dense1_baises = dense1.biases.copy()
#         best_dense2_weights = dense2.weights.copy()
#         best_dense2_baises = dense2.biases.copy()
#         lowest_loss = loss
#     else: 
#         # print("Old set of weights found, itteration: ", i," loss:", loss, " acc: ", accuracy)
#         dense1.weights = best_dense1_weights.copy()
#         dense1.biases = best_dense1_baises.copy()
#         dense2.weights = best_dense2_weights.copy()
#         dense2.biases = best_dense2_baises.copy()

# plt.plot(Loss_over_itt[1:])
# plt.ylabel('Loss (CCE)')
# plt.plot(costs)
# plt.show()



X, y = ds.create_data(100, 3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax_Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = activation2.forward(dense2.output, y)

print(activation2.output[:5])
print('loss: ', loss)

predicitons = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predicitons==y)

print('accuracy:', accuracy)

#Backwards Pass
activation2.backward(activation2.output, y)
dense2.backward(activation2.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print Gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)