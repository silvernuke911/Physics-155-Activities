def kahansum(xs):
    s,e=0,0
    for x in xs:
        temp = s
        y = x + e
        s = temp + y
        e = (temp - s) + y
    return s

print(kahansum([0.1,0.2]))