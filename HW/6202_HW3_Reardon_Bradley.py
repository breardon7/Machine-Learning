#/ Reardon, Bradley
#/ 6202 HW3

import numpy as np
import matplotlib.pyplot as plt
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
The function is decreasing when -9 < x < 1 and is concave up when x > 1.
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
critical points: 1 (local min), -9 (local max)
'''

'''
//scrap work//
f(x) = 2x^3+24x^2−54x
f prime   = 6x^2+48x-54
     = x^2+8x-9
     = (x-1)(x+9)
     x = 1, -9
     
Global min on [-3,3]: 1
Global max on [-3,3]: -9

Global min on [-infinity,0]: 1
Global max on [-infinity,0]: no max
'''
values = [float('inf')*-1, -9, -.00000000000001, 1]

for x in values:
    print(2 * x ** 3 + 24 * x ** 2 - 54 * x)

plt.plot(x2,y2)
plt.xlim(-3,3)
plt.ylim(-30,0)
plt.title('E.3.1')
plt.show()

plt.plot(x2,y2)
plt.xlim(-100000000000000,0)
#plt.ylim(-30,0)
plt.title('E.3.2')
plt.show()

#------------------------------------------------------
#E.4
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
x = np.array([1,2,0])
y = np.array([2,1,0])
z4 = x**2 + y**2
plt.plot(z4)
plt.title('E.4')
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
x = np.arange(-20,20)
y = np.arange(-20,20)
z = 2*x*y + x**2 + y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)




'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
title = ax.set_title("plot_surface: given X, Y and Z as 2D:")
title.set_y(1.01)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
fig.tight_layout()'''

#------------------------------------------------------
#E.6 (answer contained within triple quote)
#------------------------------------------------------

'''
i. f(x) = 3x - 0.5
ii. f(x) = 3x - 4
iii. f(x) = -x/5 + 2.6
iv. f(x) = -x + 3
v. f(x) = x - 2
'''

#------------------------------------------------------
#E.7
#------------------------------------------------------

'''
i. eigenvalues: 2,5
   eigenvectors: [[1,0], [0,1]]


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
   eigenvector = [0,1]
   
   check: [0,5] = [0,5]
   
   L = 2;
   [2-2, 0]   [0,0]
   [0, 2-5] = [0,-3]
   0x1 + 0x2 = 0
   0x1 = 0x2
   x1 = 1
   x2 = 0
   eigenvector = [1,0]
   
   check: [2,0] = [2,0]
   
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
    eigenvector = [1,2]
    
    check: [7, 14], [7, 14]
    
    L =3;
    [5-3, 1]   [2,1]
    [4, 5-3] = [4,2]
    2(x1) + 1(x2) = 0
    -2x1 = x2
    x1 = 1
    x2 = -2
    eigenvector = [1,-2]
    
    check: [3,-6] = [3,-6]

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
     eigenvector = [1,0.6]
     
     check: [6,3.6] = [6, 3.6]
     
     L = -2;
     [3-(-2), 5]   [5,5]
     [3, 1-(-2)] = [3,3]
     5x1 + 5x2 = 0
     -5x1 = 5x2
     x1 = 1
     x2 = -1
     eigenvector = [1,-1]
     
     check: [-2, 2] = [-2,2]

'''
v71 = np.matrix('2,0; 0,5')
v72 = np.matrix('5,1; 4,5')
v73 = np.matrix('3,5; 3,1')

print('i. eiganvalues:', np.linalg.eig(v71))
print('ii. eiganvalues:', np.linalg.eig(v72))
print('iii. eiganvalues:', np.linalg.eig(v73))