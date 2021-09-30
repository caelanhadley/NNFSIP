# from Model_1.model_1_m import Layer_Dense
import numpy as np
import math

layer_outputs = [[4.8, 1.21, 2.386],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

E = math.e

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)