import numpy as np
import nnfs
import matplotlib.pyplot as plt

nnfs.init()

def create_data(points, classes):
    X = np.zeros((points*classes, 2)) # data matrix (each row = single example)
    y = np.zeros(points*classes, dtype='uint8') # class labels
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0,1,points) # radius
        t = np.linspace(class_number*4,(class_number+1)*4,points) + np.random.randn(points)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_number
    return X, y

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    # Forward Pass
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

X, y = create_data(100, 3)
dense1 = Layer_Dense(2,3)
dense1.forward(X)

activation1 = Activation_ReLU()
activation1.forward(dense1.output)

print(activation1.output[:5])