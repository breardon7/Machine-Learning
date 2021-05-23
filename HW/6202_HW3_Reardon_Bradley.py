#/ Reardon, Bradley
#/ 6202 HW3

import numpy as np
import matplotlib.pyplot as plt

#E.1 (answer contained within triple quote)
'''
i. 6cos(6x-1)
ii. 7x^7-4/x^5
iii.((-1/x^2) – (2/x^3))e^((1/x) + (1/x^2))
iv. 12cos(6x-1)sin(6x-1)
'''

#E.2
'''
The function is decreasing when x < 0 and is concave up when x > 0.
'''
x2 = np.arange(-100,100)
y2 = 2*x2**3+24*x2**2-54*x2
plt.plot(x2,y2)
plt.show()

#E.3
'''
critical points: -1, 9
'''

f(x) = 2x^3 +24x^2 −54x
d = 2x(x^2 + 12x - 27)
  = 2x(x+1)(x-9)