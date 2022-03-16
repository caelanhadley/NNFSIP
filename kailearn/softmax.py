import numpy as np

layer_outputs = np.array([[4.8, 1.21, 2.385],
                            [8.9, -1.81, 0.2],
                            [1.41, 1.051, 0.026]])


# E = 2.71828182846
# exp_values = []
# for output in layer_outputs:
#     exp_values.append(E ** output)
# print(exp_values)

exp_values = np.exp(layer_outputs)
print(exp_values)
print()

norm_values = exp_values / np.sum(exp_values)
print(norm_values)
print()

probabilities = np.sum(norm_values, axis=1, keepdims=True)
print(probabilities)