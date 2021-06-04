#// Reardon, Bradley
#// 6202 HW5

# E1

# f(x) = x^4 - (x^3)/2 + 1

# i.
'''
f'(x) = 4x^3 -(3x^2)/2

4x^3 -(3x^2)/2 = 0
x^2(4x-3/2) = 0

x^2 = 0
x = 0

4x-3/2 = 0
8x - 3 = 0
8x = 3
x = 3/8

'''

# ii.
'''
f(x) = x^4 - (x^3)/2 + 1

(0)^4 - ((0)^3)/2 + 1 = 1
(3/8)^4 - ((3/8)^3)/2 + 1 = 0.993408203125 


'''
print((3/8)**4 - ((3/8)**3)/2 + 1)
'''
x = 3/8 = minimum
no maximum
'''

# iii.
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,1)
y = x**4 - x**3/2 + 1
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) vs x')
plt.show()
