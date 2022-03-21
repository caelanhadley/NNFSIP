from time import process_time
import numpy as np
import math


t = process_time()
#############################################################################

input = np.random.randint(0,1000,(30000000,3))
normalized = input / np.sum(input, axis=1, keepdims=True)
one_hot = np.argmax(normalized, axis=1)
loss = np.mean(-np.log(normalized[range(len(normalized)), one_hot]))
print (loss)


#############################################################################

elapsed_time = process_time() - t
print(elapsed_time)