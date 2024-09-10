# 1.1
def fact(n):
    p = 1
    if n < 0 or isinstance(n, float):
        raise ValueError('Negative or floating point numbers undefined')
    if n == 0:
        return 1
    for i in range(1,n+1):
        p*=i
    return p
print(fact(1))
