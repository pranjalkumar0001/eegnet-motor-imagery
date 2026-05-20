import math

def sigmoid(x):
    return (1/(1+math.exp(-x)))

def neuron(inputs, weights, bias):
    z = sum(i*w for i,w in zip(inputs, weights)) + bias
    return sigmoid(z)

def mse_loss(predicted, target):
    n = len(predicted)
    total_loss = sum((p-t)**2 for p,t in zip(predicted, target))
    return total_loss/n

#test
inputs  = [1.0, 2.0, 3.0]
target  = [1.0]
weights = [0.5, -0.3, 0.8]
bias    = 0.1

predicted = neuron(inputs, weights, bias)
cost = mse_loss([predicted], target)
print(f"mse_loss is {cost}")