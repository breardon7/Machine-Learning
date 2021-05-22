import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#--------------------------------------------------------
# 1
#--------------------------------------------------------

#i.
a1 = sum(np.arange(20,36)**2)
print(a1)
x1 = np.matrix('2 -4 9 -8; -3 2 -1 0; 5 4 -3 3')
print(x1)
# absolute values
print(np.abs(x1))
# squares of each element
print(np.square(x1))
# swap rows
x1[[0,1]] = x1[[1,0]]
print(x1)
# replace rows
x1[0] = 0
x1[2] = 1
# mean
print(np.mean(x1[0]))
print(np.mean(x1[2]))
# st.dev
print(np.std(x1[0]))
print(np.std(x1[2]))
# sum columns
print(sum(np.sum(x1, axis=1)))

#--------------------------------------------------------
# 2
#--------------------------------------------------------

x2 = np.arange(1,6,.1)

plt.plot(x2)
plt.xlabel('x')
plt.ylabel('x')
plt.title('x')
plt.show()

plt.plot(np.sin(x2))
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('sin(x)')
plt.show()

plt.plot(np.exp(x2))
plt.xlabel('x')
plt.ylabel('exp(x)')
plt.title('exp')
plt.show()

plt.plot(np.log(x2))
plt.xlabel('x')
plt.ylabel('log(x)')
plt.title('log(x)')
plt.show()

#--------------------------------------------------------
# 3
#--------------------------------------------------------

# i
x3 = np.random.normal(0,1, size = 1000)
y3 = np.random.uniform(0,1, size = 1000)
print(np.mean(x3))
print(np.std(x3))
print(np.mean(y3))
print(np.std(y3))

# ii
plt.hist(x3, bins=30)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Normal Dist')
plt.show()

plt.hist(y3, bins=30)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Uniform Dist')
plt.show()

#--------------------------------------------------------
# 4
#--------------------------------------------------------

def FtoC(x):
    return (x - 32)*5/9

print(FtoC(100))

#--------------------------------------------------------
# 5
#--------------------------------------------------------

def factorial(x):
    z = 1
    for i in range(x+1):
        if i > 0:
            z = z*i
    return z

print(factorial(10))

#--------------------------------------------------------
# 6
#--------------------------------------------------------

a6 = np.array([1,1,2,2,3,4,5,5,5])

def find_dups(x):
    l = list(x)
    y = []
    for i in range(len(l)):
        if l.count(l[i]) > 1:
            if l[i] not in y:
                y += [l[i]]
    return y
print(find_dups(a6))

#--------------------------------------------------------
# 7
#--------------------------------------------------------

a7 = np.array([1,4,2,3])

def second_largest(x):
    l = list(a7)
    l.sort()
    return l[-2]
print(second_largest(a7))

#--------------------------------------------------------
# 8
#--------------------------------------------------------
x8 = []
for i in range(1000):
    if i % 2 == 0:
        if i % 5 == 0:
            x8 += [i]
print(x8)



#--------------------------------------------------------
# 9
#--------------------------------------------------------

l1 = [1,2,3,4,5]
l2 = [5,6,7,8,9]
l3 = [9]

def same_or_not(x,y):
    for i in x:
        for j in y:
            if i == j:
                return "True"

print(same_or_not(l1,l2))
print(same_or_not(l1,l3))
print(same_or_not(l2,l3))

#--------------------------------------------------------
# 10
#--------------------------------------------------------

l = ['a','b','c','d','e']
def shuffle_list(x):
    from random import shuffle
    shuffle(x)
    return x
print(shuffle_list(l))
