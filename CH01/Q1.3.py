# 1.3
def print_list(list_):
    n = len(list_)
    for i in range(n-1,-1,-1):
        print(i,list_[i])
print_list([1,2,3,4,5,6])