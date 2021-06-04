import numpy as np
import matplotlib.pyplot as plt
def perceptron_learning_rule(input_vectors, targets):
    no_of_inputs = 2  # rabbit or bears
    weights = np.zeros(no_of_inputs)
    bias = np.zeros(1)
    epochs = 200  # iterations
    epochs_count = 0
    learning_rate = 0.1
    error = []
    for i in range(epochs):
        for inputs, target in zip(input_vectors, targets):
            rule = np.dot(inputs, weights) + bias  # n = w.p + b
            if rule > 0:
                activation = 1
            else:
                activation = 0
            error.append(target - activation)  # e[i] = t[i] - a
            weights += learning_rate * (target - activation) * inputs
            bias += learning_rate * (target - activation)
            epochs_count += 1
    plt.plot(np.array([0, weights[0]]), np.array([weights[1], 0]), 'go-')
    plt.xlabel("Weight on X")
    plt.ylabel("Weight on Y")
    plt.show()
    m = -weights[0] / weights[1]
    x = np.linspace(-5, 5)
    y = m * x - bias / weights[1]
    for i in range(len(targets)):
        plt.plot(input_vectors[i][0], input_vectors[i][1], 'ro' if (targets[i] == 1.0) else 'bo')
    plt.plot(x, y, 'g-')
    plt.show()
    print("The decision boundary is of weight: ", weights)
# First Trial
input_vectors = [np.array([1, 4]), np.array([1, 5]),
                 np.array([2, 4]), np.array([2, 5]), np.array([3, 1]),
                 np.array([3, 2]), np.array([4, 1]), np.array([4, 2])]
targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])
perceptron_learning_rule(input_vectors, targets)
# Second Trial
input_vectors2= [np.array([1, 3]), np.array([1, 4]), np.array([1, 5]),
                 np.array([2, 4]), np.array([2, 4]), np.array([2, 5]),
                 np.array([3, 1]), np.array([3, 2]), np.array([3, 3]), np.array([3, 4]),
                 np.array([4, 1]), np.array([4, 2]), np.array([4, 3]), np.array([4, 4])]
targets2 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
perceptron_learning_rule(input_vectors2, targets2)
