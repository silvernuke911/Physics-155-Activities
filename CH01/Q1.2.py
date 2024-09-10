# 1.2
def descr(f,x):
    functions = {
        'min' : min,
        'max' : max,
        'sum' : sum,
        'abs' : abs
    }
    func = functions.get(f,f)
    print("Function name:", f)
    print("Argument:", x)
    print("Result:", func(x))

descr('sum',[1,2,3,5])

def kahansum(xs):
    s,e=0,0
    for x in xs:
        temp = s
        y = x + e
        s = temp + y
        e = (temp - s) + y
    return s

print(kahansum([0.1,0.2]))