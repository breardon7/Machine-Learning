#/ Reardon, Bradley
#/ 6202 HW3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#------------------------------------------------------
#E.1 (answer contained within triple quote)
#------------------------------------------------------
'''
i. 6cos(6x-1)
ii. 7x^7-4/x^5
iii.((-1/x^2) – (2/x^3))e^((1/x) + (1/x^2))
iv. 12cos(6x-1)sin(6x-1)
'''

#------------------------------------------------------
#E.2 (partial answer contained within triple quote)
#------------------------------------------------------
'''
The function is decreasing when -9 < x < 1.

concave up: (-4, infinity)
f(x) = 2x^3 +24x^2 −54x
d' = 6x^2 + 48x - 54
d'' = 12x + 48 = 0
      12x = -48
      x = -4
'''
x2 = np.linspace(-20,10, 100)
y2 = 2*x2**3+24*x2**2-54*x2
plt.plot(x2,y2)
plt.title('E.2')
plt.show()

#------------------------------------------------------
#E.3 (answer contained within triple quote)
#------------------------------------------------------
'''
critical points: 1, -9

f(x) = 2x^3+24x^2−54x
f prime   = 6x^2+48x-54
     = x^2+8x-9
     = (x-1)(x+9)
     x = 1, -9

local min: (1,-28)
    2(1)^3+24(1)^2−54(1) = -28
    
local max: (-9, 972)
    2(-9)^3+24(-9)^2−54(-9) = 972
     
Global min on [-3,3]: (1, -28)
Global max on [-3,3]: (-3, 324)

Global min on [-infinity,0]: no min
Global max on [-infinity,0]: (-9, 972)
'''
#Global min/max [-3,3]
values = [-3, 1, 3]
for x in values:
    print(2 * x ** 3 + 24 * x ** 2 - 54 * x)

plt.plot(x2,y2)
plt.xlim(-20,10)
plt.ylim(-30,1000)
plt.title('E.3')
plt.show()


#------------------------------------------------------
#E.4 (partial answer contained within triple quote)
#------------------------------------------------------
'''
i. 
d/dx: 2x
d/dy: 2y

gradient vector: [2x, 2y]

ii.
@ (1,2): [2, 4]
@ (2,1): [4, 2]
@ (0,0): [0, 0]
'''

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = X**2 + Y**2

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('X^2 + Y^2')
plt.show()


#------------------------------------------------------
#E.5 (partial answer contained within triple quote)
#------------------------------------------------------
'''
i. 
d/dx: 2x + 2y
d/dy: 2x + 1

gradient vector: [2x + 2y, 2x + 1]

ii.
@ (1,1): [4, 3]
@ (0,-1): [-2, 1]
@ (0,0): [0, 1]
'''

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = 2*X*Y + X**2 + Y

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('X^2 + 2XY + Y')
plt.show()


#------------------------------------------------------
#E.6 (answer contained within triple quote)
#------------------------------------------------------

'''
i. f(x) = 3x - 0.5
    -0.5 = 3(0) + b
    b = -0.5
ii. f(x) = 3x - 4
    m = (14-8)/(6-4) = 3
    8 = 3(4) + b
    b = -4
iii. f(x) = -x/5 + 2.4
    m = -1/5
    2 = -2/5 + b
    12/5 = b
    b = 2.4
iv. f(x) = -x + 3
    1 = m(2) + 3
    -2 = 2m
    m = -1
v. f(x) = x - 2
    m = (-1-4)/(1-6) = 1
    4 = 6 + b
    b = -2
'''

#------------------------------------------------------
#E.7 (partial answer contained within triple quote)
#------------------------------------------------------

'''
i. eigenvalues: 2,5
   eigenvectors: [[1 0], [0 1]]


   [2-L, 0]
   [0, 5-L]
   L^2 - 7L + 10 = 0
   (L-5)(L-2)
   L = 5, 2
   
   L = 5;
   [2-5, 0]   [-3,0]
   [0, 5-5] = [0,0]
   -3x1 + 0x2 = 0
   -3x1 = 0x2
   x1 = 0
   x2 = 1
   eigenvector = [0 1]
   
   check: 5 * [0 1] = [0 5] 
          [2 0] * [0 1] = [0 5]
          [0 5]
   L = 2;
   [2-2, 0]   [0,0]
   [0, 2-5] = [0,-3]
   0x1 + 0x2 = 0
   0x1 = 0x2
   x1 = 1
   x2 = 0
   eigenvector = [1 0]
   
   check: 2* [1 0] = [2 0] = [2 0]
             [2 0] * [1 0] = [2 0]
             [0 5]
ii. eigenvalues: 7, 3
    eigenvectors: [[1,2], [1,-2]]

    [5-L, 1]
    [4, 5-L]
    L^2 - 10L + 21
    (L-3)(L-7)
    L = 7, 3
    
    L =7;
    [5-7, 1]   [-2,1]
    [4, 5-7] = [4,-2]
    -2(x1) + 1(x2) = 0
    2x1 = x2
    x1 = 1
    x2 = 2
    eigenvector = [1 2]
    
    check:     7 * [1 2] = [7 14]
           [5 1] * [1 2] = [7 14]
           [4 5]
    
    L = 3;
    [5-3, 1]   [2,1]
    [4, 5-3] = [4,2]
    2(x1) + 1(x2) = 0
    -2x1 = x2
    x1 = 1
    x2 = -2
    eigenvector = [1,-2]

    check:     3 * [1 -2] = [3 -6]
           [5 1] * [1 -2] = [3 -6]
           [4 5]

iii. eigenvalues: 6, -2
     eigenvectors: [[1, 0.6], [1,-1]]


     [3-L, 5]
     [3, 1-L]
     L^2 - 4L - 12
     (L-6)(L+2)
     L = 6, -2
     
     L = 6;
     [3-6, 5]   [-3,5]
     [3, 1-6] = [3,-5]
     -3x1 + 5x2 = 0
     3x1 = 5x2
     x1 = 1
     x2 = 0.6
     eigenvector = [1 0.6]
     
    check:     6 * [1 0.6] = [6 3.6]
           [3 5] * [1 0.6] = [6 3.6]
           [3 1]

     
     L = -2;
     [3-(-2), 5]   [5,5]
     [3, 1-(-2)] = [3,3]
     5x1 + 5x2 = 0
     -5x1 = 5x2
     x1 = 1
     x2 = -1
     eigenvector = [1 -1]
     
    check:    -2 * [1 -1] = [-2 2]
           [3 5] * [1 -1] = [-2 2]
           [3 1]     


'''
v71 = np.matrix('2,0; 0,5')
v72 = np.matrix('5,1; 4,5')
v73 = np.matrix('3,5; 3,1')

print('i. eig:', np.linalg.eig(v71))
print('ii. eig:', np.linalg.eig(v72))
print('iii. eig:', np.linalg.eig(v73))