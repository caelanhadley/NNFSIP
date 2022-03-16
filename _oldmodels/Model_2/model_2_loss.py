import math

softmax_output = [0.7, 0.1, 0.2]

'''
One-Hot Encoding
'''
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]*target_output[0] +
                    softmax_output[1]*target_output[1] +
                    softmax_output[2]*target_output[2]))

print(loss)

loss = -math.log(softmax_output[0])

print(loss)