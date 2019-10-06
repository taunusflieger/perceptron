import numpy as np
import csv
import matplotlib.pyplot as plt

X = np.array(([],[]))
y = np.array([])

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.0001):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines


# Load training dataset
with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
            p = np.array([(float)(row[0]),(float)(row[1])])
            if line_count == 0:
                X = p
                y = np.array([(float)(row[2])])
            else:
                X = np.vstack((X, p))
                y = np.vstack((y, np.array([(float)(row[2])])))
            line_count += 1

bl = trainPerceptronAlgorithm(X, y)

# Set X and Y axis limits
plt.ylim(-0.5, 1.5)
plt.xlim(-0.5, 1.5)
for i in range(len(X)):
    if y[i] == 1:
        plt.scatter(X[i][0], X[i][1], marker='o', c='b', edgecolor='black')
    else:
        plt.scatter(X[i][0], X[i][1], marker='o', c='r', edgecolor='black')

i = 1
for x1, y1 in bl:
    x_line = np.linspace(np.min(X), np.max(X), 10)
    y_line = x1 * x_line + y1

    if i < len(bl):
        color_setting = 'green'
        linestyle_setting = '--'
    else:
        # Final line will be drawn differently
        color_setting = 'black'
        linestyle_setting = '-'
    plt.plot(x_line,y_line, linestyle=linestyle_setting, color=color_setting)
    i += 1
plt.show()