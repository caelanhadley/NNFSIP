# This is a sample neuron with three inputs
# To Demonstrate Forward and backward propagation
# on the scale of a single neuron.

x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Forward Pass
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2] 
print(xw0, xw1, xw2, b)
# sum
z = xw0 + xw1 + xw2 + b
print(z)
# Relu
y = max(z, 0)
print(y)

# Backwards Pass

# Derivative from the next layer
dvalue = 1.0

# Derivative of Relu
dy = dvalue * (1 if y > 0 else 0)
# >>> y = 6; dy = 1

# Derivative of :
# Sum
dsum = 1

# Multiplication
dmul_dx0 = w[0]

# The Entire Partial Derivative of ReLU with respect to x0 (steps)
drelu_dx0 = w[0]
drelu_dx0 = w[0] * 1
drelu_dx0 = w[0] * 1 * dvalue * (1 if y > 0 else 0)
drelu_dx0 = w[0] * dvalue * (1 if y > 0 else 0)

# My Back Prop Algo
dx = []
dw = []
db = dvalue * (1 if y > 0 else 0) * 1
for weight, input in zip(w,x):
    dx.append(weight * dvalue  * (1 if y > 0 else 0))
    dw.append(input * dvalue * (1 if y > 0 else 0))
print(dx, dw, db)

import numpy as np

alpha = 0.001
print(w)
w += -alpha * np.array(dw)
b += -alpha * db
print(w)

# Second Forward pass
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2] 
print(xw0, xw1, xw2, b)
# sum
z = xw0 + xw1 + xw2 + b
print(z)
# Relu
y = max(z, 0)
print(y)