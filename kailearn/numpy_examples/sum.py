import numpy as np

input = np.array([[4.8, 1.21, 2.385],
                    [8.9, -1.81, 0.2],
                    [1.41, 1.051, 0.026]])

print("Raw Input")
print(input)
print("\nSum axis=0")
print(np.sum(input, axis=0, keepdims=True))
print("\nSum axis=1")
print(np.sum(input, axis=1, keepdims=True))