#// Reardon, Bradley
#// 6202 HW5

#E.1
import numpy as np
import matplotlib.pyplot as plt
S1 = 2
input = np.arange(-2,3)
lr = 0.15
w = np.random.rand(S1, len(input))
b = np.random.rand(len(input), 1)
print(w,b)