import numpy as np

traindata = np.loadtxt("train.csv", delimiter=",", skiprows=1)


trainPixels = traindata[:, 1:] / 255
trainLabels = traindata[:, 0].astype(int)


input_size = 784
hidden1_size = 32
hidden2_size = 16
output_size = 10
lr = 0.01
epochs = 100
batch_size = 128


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deri(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    z = x - np.max(x, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def makeAnsArray(labels):
    one_hot = np.zeros((labels.size, 10))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def costFuncRes(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred), axis=1)


# FORWARD PROPOGATION CODE
# INITIALISES RANDOM WEIGHTS AND BIASES
###############################################

# between inp lay and 32 neuron lay

# randomWeightsL1 = np.random.uniform(-1, 1, size=(input_size, hidden1_size))
# randomBiasL1 = np.random.uniform(-1, 1, size=(hidden1_size))
# z1 = trainPixels @ randomWeightsL1 + randomBiasL1
# a1 = sigmoid(z1)
# layer32neuron =  sigmoid(trainPixels @ randomWeightsL1 + randomBiasL1)

# # bw 32 lay and 16 lay
# randomWeightsL2 = np.random.uniform(-1, 1, size=(hidden1_size, hidden2_size))
# randomBiasL2 = np.random.uniform(-1, 1, size=(hidden2_size))
# z2 = a1 @ randomWeightsL2 + randomBiasL2
# a2 = sigmoid(z2)
# layer16neuron = sigmoid(layer32neuron @ randomWeightsL2  + randomBiasL2)

# # between 16 lay and out lay
# randomWeightsL3 = np.random.uniform(-1, 1, size=(hidden2_size, output_size))
# randomBiasL3 = np.random.uniform(-1, 1, size=(output_size))

# z3 = a2 @ randomWeightsL3 + randomBiasL3
# a3 = softmax(z3)
# outputNeuron = softmax(layer16neuron @ randomWeightsL3 + randomBiasL3)

# print(outputNeuron.shape, layer16neuron.shape, layer32neuron.shape, trainPixels.shape)
# np.savetxt("outputinitialrandom.csv", outputNeuron, delimiter=',')




randomWeightsL1 = np.random.uniform(-1, 1, size=(input_size, hidden1_size))
randomBiasL1    = np.random.uniform(-1, 1, size=(hidden1_size))

randomWeightsL2 = np.random.uniform(-1, 1, size=(hidden1_size, hidden2_size))
randomBiasL2    = np.random.uniform(-1, 1, size=(hidden2_size))

randomWeightsL3 = np.random.uniform(-1, 1, size=(hidden2_size, output_size))
randomBiasL3    = np.random.uniform(-1, 1, size=(output_size))


answerarray = makeAnsArray(trainLabels)

# costArr = costFuncRes(answerarray, outputNeuron)
# # print(trainLabels)
# # print(costArr)
# print(np.average(costArr))


# BACK PROPOGATION CODE
# CHANGE WEIGHTS AND BIASES USING GRADIENT DESCENT
######################################################


# chain rule for mean squared error
# deri3 = 2 * (outputNeuron - answerarray)

# gradWtsL3 = layer16neuron.T @ deri3
# gradBiasL3 = np.sum(deri3, axis=0)


# deri2 = (deri3 @ randomWeightsL3.T) * sigmoid_deri(z2)
# gradWtsL2 = a1.T @ deri2
# gradBiasL2 = np.sum(deri2, axis=0)

# deri1 = (deri2 @ randomWeightsL2.T) * sigmoid_deri(z1)
# gradWtsL1 = trainPixels.T @ deri1
# gradBiasL1 = np.sum(deri1, axis=0)

# randomWeightsL3 -= lr * gradWtsL3
# randomBiasL3    -= lr * gradBiasL3

# randomWeightsL2 -= lr * gradWtsL2
# randomBiasL2    -= lr * gradBiasL2

# randomWeightsL1 -= lr * gradWtsL1
# randomBiasL1    -= lr * gradBiasL1

for epoch in range(epochs):
    # Shuffle the data
    indices = np.arange(trainPixels.shape[0])
    np.random.shuffle(indices)
    trainPixels = trainPixels[indices]
    answerarray = answerarray[indices]
    
    total_loss = 0
    for i in range(0, trainPixels.shape[0], batch_size):
        X_batch = trainPixels[i:i+batch_size]
        Y_batch = answerarray[i:i+batch_size]
        
        # --- FORWARD ---
        z1 = X_batch @ randomWeightsL1 + randomBiasL1
        a1 = sigmoid(z1)
        
        z2 = a1 @ randomWeightsL2 + randomBiasL2
        a2 = sigmoid(z2)
        
        z3 = a2 @ randomWeightsL3 + randomBiasL3
        a3 = softmax(z3)
        
        # --- LOSS ---
        loss = costFuncRes(Y_batch, a3)
        total_loss += np.sum(loss)

        # --- BACKPROP ---
        d3 = 2 * (a3 - Y_batch)  # MSE derivative

        gradW3 = a2.T @ d3
        gradB3 = np.sum(d3, axis=0)

        d2 = (d3 @ randomWeightsL3.T) * sigmoid_deri(z2)
        gradW2 = a1.T @ d2
        gradB2 = np.sum(d2, axis=0)

        d1 = (d2 @ randomWeightsL2.T) * sigmoid_deri(z1)
        gradW1 = X_batch.T @ d1
        gradB1 = np.sum(d1, axis=0)

        # --- UPDATE ---
        randomWeightsL3 -= lr * gradW3
        randomBiasL3    -= lr * gradB3

        randomWeightsL2 -= lr * gradW2
        randomBiasL2    -= lr * gradB2

        randomWeightsL1 -= lr * gradW1
        randomBiasL1    -= lr * gradB1

    # Print loss at end of epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / trainPixels.shape[0]:.4f}")


np.savez("trained_model_100epochs.npz",
         W1=randomWeightsL1, B1=randomBiasL1,
         W2=randomWeightsL2, B2=randomBiasL2,
         W3=randomWeightsL3, B3=randomBiasL3)
