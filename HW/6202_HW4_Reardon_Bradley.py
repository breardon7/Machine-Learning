#// Reardon, Bradley
#// 6202 HW4

import numpy as np
import matplotlib.pyplot as plt

# i.
def hardlim(x):
    if x < 0:
        return 0
    else:
        return 1

def ppn(p,t):
    epoch = 100
    inputs = 2 # classes
    b = np.ones(1)
    w = np.ones(inputs)
    epochs = 0
    #learning_rate = 0.2
    for j in range(epoch):
        for i in range(len(p)):
            n = np.dot(w,p[i]) + b
            a = hardlim(n)
            e = t[i] - a
            w = w + e*p[i]
            b = b + e
        epochs += 1
        E[j] =


og_inputs = [np.array([1, 4]), np.array([1, 5]), np.array([2, 4]), np.array([2, 5]),
             np.array([3, 1]), np.array([3, 2]), np.array([4, 1]), np.array([4, 2]),]

og_targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])