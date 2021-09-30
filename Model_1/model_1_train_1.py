from model_1_m import *
from matplotlib import pyplot as plt

Loss_over_itt = []

X, y = cs.create_data(100, 3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CatagoricalCrossEntropy()

# Experimental #


lowest_loss = 99999

best_dense1_weights = dense1.weights.copy()
best_dense1_baises = dense2.biases.copy()
best_dense2_weights = dense1.weights.copy()
best_dense2_baises = dense2.biases.copy()

for i in range(10000):
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1,3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)
    # print(loss)

    predicitions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predicitions==y)

    Loss_over_itt.append(lowest_loss)

    if loss < lowest_loss:
        # print("New set of weights found, itteration: ", i," loss:", loss, " acc: ", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_baises = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_baises = dense2.biases.copy()
        lowest_loss = loss
    else: 
        # print("Old set of weights found, itteration: ", i," loss:", loss, " acc: ", accuracy)
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_baises.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_baises.copy()

plt.plot(Loss_over_itt[1:])
plt.ylabel('Loss (CCE)')
plt.show()