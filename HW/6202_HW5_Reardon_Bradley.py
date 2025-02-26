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


# E2
'''
point = [1 1]
direction = [-1 1]
'''
# i.
'''
f(x) = 7/2x1^2 - 6x1x2 - x2^2

first gradient  = [7x1-6x2, -6x1-2x2]
-(7(1)-6(1)) + (-6(1)-2(1)) = -9/sqrt(2)

second gradient = [7  -6]
                  [-6 -2]
[-1 1]*[7  -6]* [-1] = [-13 4] * [-1] = 17/2
       [-6 -2]  [ 1]             [ 1]

'''

# ii.
'''
f(x) = 5x1^2 - 6x1x2 + 5x2^2 + 4x1 + 4x2

first gradient  = [10x1-6x2-4, 10x2-6x1+4]
-(10(1)-6(1)-4) + (10(1)-6(1)-4) = 0

second gradient = [10 -6]
                  [-6 10]
[-1 1]*[10 -6]* [-1] = [-16 16] *[-1] = 16
       [-6 10]  [ 1]             [ 1]                 
'''

# iii.
'''
f(x) = -7/2x1^2 - 6x1x2 + x2^2

first gradient  = [-7x1-6x2, 2x2-6x1]
-(-7(1)-6(1)) + (2(1)-6(1)) = 9/sqrt(2)

second gradient = [-7, -6]
                  [-6   2]
[-1 1]*[-7 -6]* [-1] = [1 8] * [-1] = 7/2
       [-6  2]  [ 1]           [ 1]  
'''

# E3

e = np.e
x = np.linspace(-100,100)
y1 = e**1-e**1*(x-1)
y2 = e**1-e**1*(x-1)-1/2*e**1*(x-1)**2
y3 = e**1-e**1*(x-1)-1/2*e**1*(x-1)**2-1/6*e**1*(x-0)**3
labels = ['original function', 'first-order', 'second-order', 'third-order']

plt.plot(x,y1, 'b')
plt.plot(x,y2, 'r')
plt.plot(x,y3, 'y')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Taylor Series Appoximations')
plt.legend(labels)
plt.show()

# iv.
'''
As we move into higher order approximations, we move closer to the original f(x).
'''