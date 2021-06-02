#// Reardon, Bradley
#// 6202 HW4

# i.

b =
w =
p = {[1,4], []}
f =

for j in range(epoch):
    for i in range(len(p)):
        n = w*p + b
        a = f(n)
        e = t[i] - a
        w = w + e*p
        b = b + e