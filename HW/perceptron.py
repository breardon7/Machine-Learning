import numpy as np
import matplotlib.pyplot as plt
def perceptron_learning(inp, target):
    w = np.zeros(2, dtype=float)
    b = np.ones(1, dtype=float)
    observation_count = len(target)
    observation_float = float(observation_count)
    mse_values = []
    epoch_values = []
    epoch = 20
    learning_rate = 0.1
    epoch_current = 1
    for i in range(epoch):
        error_sum = 0.0
        for x in range(observation_count):
            #print('inp val is', inp[x])
            #print('w is',w)
            #print('b is',b)
            n = np.dot(inp[x], w) + b
            #print('n is',n)
            a = 1.0 if n > 0 else 0.0
            #print('a is',a)
            #print('target is', target[x])
            e = target[x] - a
            #print('e is',e)
            error_sum += float(e * e)
            #print(error_sum)
            w = w + learning_rate * e * inp[x]
            b = b + learning_rate * e
        #print('error_sum', error_sum)
        mse = float(error_sum) / observation_float
        mse_values.append(mse)
        epoch_values.append(epoch_current)
        epoch_current += 1
    # Test final weights and bias against the input data
    # Every datapoint seems to be accurate
    for x in range(observation_count):
        n = np.dot(inp[x], w) + b
        a = 1.0 if n > 0 else 0.0
        print(f'The output is {int(a)} and the target is {target[x]}')
    # Plot error vs epochs
    plt.plot(epoch_values, mse_values)
    plt.xlabel('Epoch')
    plt.ylabel('Error (MSE)')
    plt.title('Error (MSE) vs Epoch')
    plt.show()
    # Plot decision boundary
    #Turn weights and bias into a line
    #plot the line
    #plot each point with the color for its classification
inp = [np.array([1,4]), np.array([1,5]), np.array([2,4]), np.array([2,5]),
       np.array([3,1]), np.array([3,2]), np.array([4,1]), np.array([4,2])]
target = [0, 0, 0, 0, 1, 1, 1, 1]
perceptron_learning(inp, target)