# 1.4
def der(f, x, h=0.01):
    return (f(x+h) - f(x))/h
def fa(x):
    return x**2
def fb(x):
    return x+1
def caller(a, b):
    def func(x):
        return a * fa(x) + b * fb(x)
    return func
a,b = 1,10
x = 2
y = der(caller(a,b),x)
print(x,y)