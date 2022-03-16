import numpy as np

input = np.random.randint(0,1000,(3,3))
normalized = input / np.sum(input, axis=1, keepdims=True)

print(normalized)
print(np.argmax(normalized, axis=1))

    