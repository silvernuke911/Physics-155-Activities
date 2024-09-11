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

print(2**32)