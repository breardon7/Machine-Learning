#// Reardon, Bradley
#// 6202 HW4

import numpy as np
import matplotlib.pyplot as plt


def hardlim(x):
    if x < 0:
        return 0
    else:
        return 1

def ppn(p,t):
    epoch = 100000
    inputs = 2 # classes
    b = np.ones(1)
    w = np.ones(inputs)
    lr = 0.1
    #learning_rate = 0.2
    for j in range(epoch):
        for i in range(len(p)):
            n = np.dot(w,p[i]) + b
            a = hardlim(n)
            e = t[i] - a
            w = w + lr*e*p[i]
            b = b*lr + e
    for i in range(len(p)):
        plt.plot(p[i][0], p[i][1], 'rx' if (t[i] == 1) else 'bx')
    x = np.linspace(0, 5)
    m = -w[0]/w[1]
    y = m*x + b
    plt.plot(x,y)
    plt.show()


# Initial Train
og_inputs = [np.array([1, 4]), np.array([1, 5]), np.array([2, 4]), np.array([2, 5]),
             np.array([3, 1]), np.array([3, 2]), np.array([4, 1]), np.array([4, 2])]

og_targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])

ppn(og_inputs, og_targets)

# Second Train
new_inputs = [np.array([1, 4]), np.array([1, 5]), np.array([2, 4]), np.array([2, 5]), np.array([1, 2]), np.array([3, 5]), np.array([3, 6]), np.array([4, 6]),
             np.array([3, 1]), np.array([3, 2]), np.array([4, 1]), np.array([4, 2]), np.array([2, 1]), np.array([1, .5]), np.array([2.5, 2]), np.array([5, 4])]

new_targets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

ppn(new_inputs, new_targets)

