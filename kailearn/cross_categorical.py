import numpy as np
import math

input = np.random.randint(0,1000,(3,3))
# input = input * [[1, 200, 1],[1, 1, 200],[200, 1, 1]]
normalized = input / np.sum(input, axis=1, keepdims=True)
one_hot = np.zeros(input.shape)

for i in range(normalized.shape[0]):
    one_hot[i, np.argmax(normalized[i])] = 1
    print(one_hot)

print("Normalized Matrix")
print(normalized)
print("\nOne-Hot")
print(one_hot)
print("\nArgmax (Normalized Matrix)")
print(np.argmax(normalized))
print("\nOut Values")
print()



total_loss = np.sum(
    np.log(
        np.sum(normalized * one_hot, axis=1, keepdims=True)
    )
)

output = normalized * one_hot
print(output)
print()
print (-total_loss)
    