# =====================================================================
# PROBLEM: Calibrating a Joint Sensor (Quadratic Regression)
# =====================================================================
# Context: You are calibrating a millimeter-scale sensor for a 
# robotic arm joint. The relationship between the raw sensor reading (x) 
# and the actual physical angle (y) is non-linear and follows a curve.
# 
# Model Equation: y = (w2 * x**2) + (w1 * x) + b
#
# Task: Write a gradient descent script to find the optimal weights 
# (w2, w1) and bias (b).
#
# The Dataset:
# x_inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
# y_targets = [11.0, 4.0, 1.0, 2.0, 7.0]
#
# Target Check: 
# The perfect fit is w2 = 2.0, w1 = -1.0, b = 1.0
# =====================================================================

# Write your code below:

def mse_loss(predicted, target):
    total_loss = sum((p-t)**2 for p,t in zip(predicted, target))
    n = len(predicted)
    return total_loss/n

def grad_desc(x_input, y_target):
    w1 = 0
    w2 = 0
    b = 0
    n = len(y_target)
    learning_rate = 0.0001
    for epoch in range(100000):
        predicted = [((w2*x**2) + (w1*x) + b) for x in x_input]
        dw2 = sum((2*(p-y)*(x**2)) for p,y,x in zip(predicted, y_target, x_input))/n
        dw1 = sum((2*(p-y)*x) for p,y,x in zip(predicted, y_target, x_input))/n
        db = sum((2*(p-y)) for p,y in zip(predicted, y_target))/n
        w1 -= dw1 * learning_rate
        w2 -= dw2 * learning_rate
        b -= db * learning_rate
        #if (epoch % 100 == 0):
         #   print(f"for epoch {epoch} mse_loss is {mse_loss(predicted, y_target)}")
    
    print(f"final values are w2={w2}, w1={w1}, b={b}")

input = [-2.0, -1.0, 0.0, 1.0, 2.0]
target = [11.0, 4.0, 1.0, 2.0, 7.0]
grad_desc(input, target)
