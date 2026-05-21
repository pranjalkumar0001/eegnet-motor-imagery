# =====================================================================
# PROBLEM: System Identification of a Damped Pendulum
# =====================================================================
# Context: You recorded the angle (theta) of a swinging pendulum over 
# time (t). The amplitude decays exponentially due to friction.
#
# The Governing Physics Equation:
# theta(t) = A * e^(-k * t) * cos(w * t)
#
# Where:
# A = Initial amplitude
# k = Damping coefficient
# w = Angular frequency
#
# Task:
# 1. Define an appropriate cost function to measure the error.
# 2. Derive the partial derivatives for A, k, and w entirely on paper.
# 3. Write the gradient descent algorithm to find the optimal values 
#    for A, k, and w.
#
# The Lab Data:
# t_inputs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# theta_targets = [5.00, 2.10, -1.26, -2.34, -1.20, 0.41, 1.07]
#
# Initialization Hint:
# Because of the cosine wave, a bad initialization will trap you in a 
# local minimum. Start with: A = 4.0, k = 0.0, w = 1.0
#
# Note: You will need the `math` module for math.exp() and math.cos(), 
# as well as math.sin() for your gradients.
# =====================================================================

import math
def mse_loss(predicted, target):
    return sum((p-t)**2 for p,t in zip(predicted, target))/len(predicted)

def part_derrivative(input, target, A, k, w):
    predicted = [(A*(math.exp(-k*t))*(math.cos(w*t))) for t in input]
    dA = sum((2*(p-y)*p/A) for p,y in zip(predicted, target))/len(target)
    dk = sum(2*(-t)*(p-y)*p for p,y,t in zip(predicted, target, input))/len(target)
    dw = sum(2*(-t)*(p-y)*p*(math.tan(w*t)) for p,y,t in zip(predicted, target, input))/len(target)
    return dA, dk, dw, predicted

def grad_desc(input, target):
    A = 4
    k = 0
    w = 1
    learning_rate = 0.0001
    for epoch in range(1000000):
        dA, dk, dw, predicted = part_derrivative(input, target, A, k, w)
        A -= dA*learning_rate
        k -= dk*learning_rate
        w -= dw*learning_rate
        if (epoch%100000 == 0):
            print(f"for epoch={epoch} the cost is {mse_loss(predicted, target)}")
    print(f"values of A={A}, k={k}, w={w}")

t_inputs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
theta_targets = [5.00, 2.10, -1.26, -2.34, -1.20, 0.41, 1.07]
grad_desc(t_inputs, theta_targets)