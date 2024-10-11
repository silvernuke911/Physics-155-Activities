import random as rnd
import math as mt
import matplotlib.pyplot as plt

def numerical_differentiator_backward(x_list,f_x_list):
    if len(x_list)!=len(f_x_list):
        return "variable and function do not match"
    output=[]
    for i in range(len(x_list)):
        if i-1==-1:
            m=(f_x_list[i+1]-f_x_list[i])/(x_list[i+1]-x_list[i])
        m=(f_x_list[i]-(f_x_list[i-1]))/(x_list[i]-x_list[i-1])
        output.append(m)
    return output

def numerical_differentiator_forward(x_list,f_x_list):
    if len(x_list)!=len(f_x_list):
        return "variable and function do not match"
    output=[]
    for i in range(len(x_list)):
        if i+1==len(x_list):
            m=(f_x_list[i]-f_x_list[i-1])/(x_list[i]-x_list[i-1])
        else:
            m=(f_x_list[i+1]-(f_x_list[i]))/(x_list[i+1]-x_list[i])
        output.append(m)
    return output

def numerical_differentiator_central(x_list,f_x_list):
    if len(x_list)!=len(f_x_list):
        return "variable and function do not match"
    output=[]
    for i in range(len(x_list)):
        if i-1==-1:
            m=(f_x_list[i+1]-f_x_list[i])/(x_list[i+1]-x_list[i])
        elif i+1==len(x_list):
            m=(f_x_list[i]-f_x_list[i-1])/(x_list[i]-x_list[i-1])
        else:
            m=(f_x_list[i+1]-f_x_list[i-1])/(x_list[i+1]-x_list[i-1])
        output.append(m)
    return output
    
def numerical_integrator_left(x_list,f_x_list,c=0):
    if len(x_list)!=len(f_x_list):
        return "variable and function do not match"
    output=[]
    s=0
    for i in range(len(x_list)):
        if i-1==-1:
            s=(f_x_list[i])*(x_list[i+1]-x_list[i])+c
        else:
            s+=(f_x_list[i-1])*(x_list[i]-x_list[i-1])
        output.append(s)
    return output
    
def numerical_integrator_right(x_list,f_x_list,c=0):
    if len(x_list)!=len(f_x_list):
        return "variable and function do not match"
    output=[]
    s=0
    for i in range(len(x_list)):
        if i==0:
            s+=c
        elif i+1==len(x_list):
            s+=(f_x_list[i])*(x_list[i]-x_list[i-1])
        else:
            s+=(f_x_list[i])*(x_list[i+1]-x_list[i])
        output.append(s)
    return output
    
def numerical_integrator_middle(x_list,f_x_list,c=0):
    if len(x_list)!=len(f_x_list):
        return "variable and function do not match"
    output=[]
    s=0
    def mean(n1,n2):
        return (n1+n2)/2
    for i in range(len(x_list)):
        if i==0:
            s+=c
        elif i==len(x_list):
            s+=(f_x_list[i])*(x_list[i]-x_list[i-1])
        else:
            s+=mean(f_x_list[i],f_x_list[i-1])*(x_list[i]-x_list[i-1])
        output.append(s)
    return output

def double_numerical_differentiator(x_list,f_x_list):
    if len(x_list)!=len(f_x_list):
        return "variable and function do not match"
    output1=numerical_differentiator_central(x_list,f_x_list)
    output2=numerical_differentiator_central(x_list,output1)
    return output2

def double_numerical_integrator(x_list,f_x_list,c=0):
    if len(x_list)!=len(f_x_list):
        return "variable and function do not match"
    output1=numerical_integrator_middle(x_list,f_x_list)
    output2=numerical_integrator_middle(x_list,output1)
    return output2
def newton_zero(f_x,df_x,init_guess,tol=1e-8,max_iter=100):
    x0 = init_guess
    for _ in range(max_iter):
        x1 = x0 - f_x(x0) / df_x(x0)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
    return None


#sample for newton_zero
def func(x):
    return x**2+2*x+1

def dfunc(x):
    return 2*x+2

#tester code
def time_interval(time_start,time_end,step):
    decimals=len(str(step))-1
    steps=round((time_end-time_start)/step)
    time=time_start-step
    timelist=[]
    for i in range(0,steps+1):
        time+=step
        timelist.append(round(time,decimals))
    return(timelist)

def custom_sin(time,amplitude=1,phase=0,rand=False):
    if rand==True:
        return amplitude*mt.sin(time-phase)+rnd.uniform(-0.1,1)
    return amplitude*mt.sin(time-phase)

def custom_cos(time,amplitude=1,phase=0,rand=False):
    if rand==True:
        return -amplitude*mt.cos(time-phase)+rnd.uniform(-0.1,1)
    return -amplitude*mt.cos(time-phase)

def square_wave(time,frequency=0.25,amplitude=1,phase=0,offset=0):
    x=amplitude*mt.sin(2*mt.pi*frequency*time+phase)+offset
    if x>=0:
        return amplitude
    if x<0:
        return -amplitude



def function_list_maker(x_list,f_x):
    output=[]
    for i in range(len(x_list)):
        output.append(f_x(x_list[i]))
    return output

def stringlister(*lists):
    # Combines multiple lists of strings into a single list with matching indices.
    # Args:
    # *lists (list of str): Multiple lists of strings.
    # Returns:
    # list of str: Combined list with matching indices.
    combined_list = []
    for items in zip(*lists):
        combined_list.append("".join(str(items)).replace("(", "").replace(")", ""))
    return combined_list

def file_opener_writer(filename,stringlist):
    file = open (filename, 'w')
    for _ in range(len(stringlist)):
        print(stringlist[_],end='\r')
        file.write(stringlist[_]+'\n')
    file.close() 


def average_error(list1_original,list2_experimental):
    
    def error(n0,nE):
        if n0==0:
            return 0
        return abs((n0-nE)/n0)
    n=len(list1_original)
    if len(list2_experimental)!=n:
        return "list length does not match"
    sum=0
    for i in range(n):
        sum+=error(list1_original[i],list2_experimental[i])
    return sum/n

x_min,x_max=0,10
y_min,y_max=-2.5,2.5    
def plot():
    plt.axis([x_min, x_max, y_min, y_max])
    plt.grid()
    plt.plot(
        timelist,
        function_list,
        color="red",
        marker="",
        linestyle="-"
    )
    # plt.plot(
    #     timelist,
    #     derivative_list,
    #     color="orange",
    #     marker="^",
    #     linestyle="-"
    # )
    # plt.plot(
    #     timelist,
    #     integral_list,
    #     color="blue",
    #     marker="o"
    # )
    plt.plot(
        timelist,
        dm_list,
        color="green",
        marker=""
    )
    plt.plot(
        timelist,
        d2m_list,
        color="blue",
        marker=""
    )
    plt.plot(
        timelist,
        im_list,
        color="magenta",
        marker=""
    )
    plt.plot(
        timelist,
        i2m_list,
        color="yellow",
        marker=""
    )
    plt.show()

# def func(x):
#     return x**2#+rnd.uniform(-0.1,0.1)
# def deriv_func(x):
#     return 2*x
# def integ_func(x):
#     return (1/3)*x**3

# def func(x):
#     y=0
#     if x>=0 and x<2:
#         y=3*x
#     elif x>=2 and x<7:
#         y=6
#     elif x>=7:
#         y=-2*x+20
#     return y

def func(x):
    return mt.sin(x)
# def func(x):
#     return x
# def deriv_func(x):
#     return 1
# def integ_func(x):
#     return (1/2)*x**2

# def func(x):
#     return (1/mt.sqrt(2*mt.pi))*mt.e**-((x)**2)
# def func(x):
#     return mt.sin(x)
# def deriv_func(x):
#     return mt.cos(x)
# def integ_func(x):
#     return -mt.cos(x)

timelist=time_interval(x_min,x_max,1)
function_list=function_list_maker(timelist,func)
# derivative_list=function_list_maker(timelist,deriv_func)
# integral_list=function_list_maker(timelist,integ_func)
il_list=numerical_integrator_left(timelist,function_list,0)
# ir_list=numerical_integrator_right(timelist,function_list,0)
dm_list=numerical_differentiator_central(timelist,function_list)
d2m_list=double_numerical_differentiator(timelist,function_list,)
im_list=numerical_integrator_middle(timelist,function_list,-1)
i2m_list=double_numerical_integrator(timelist,function_list,0)
# sqrlist=function_list_maker(timelist,square_wave)
# dsinlist=numerical_differentiator_backward(timelist,sinlist)
# fsinlist=numerical_differentiator_forward(timelist,sinlist)
# g_list=numerical_differentiator_central(timelist,sinlist)

flist=stringlister(timelist,function_list,dm_list,d2m_list,im_list,i2m_list)
file_opener_writer('defwav2.csv',flist)
# print('Derivative Error:',100*average_error(derivative_list,dm_list),"%")
# print('Integral Error:',100*average_error(integral_list,im_list),"%")
plot()



def r_squared(x_list,y_list,function):
    def ss_res(x_list,y_list,function):
        sum=0
        for i in range(len(x_list)):
            sum+=(y_list[i]-function(x_list[i]))**2
        return sum
    def ss_tot(y_list):
        ave=mean(y_list)
        sum=0
        for i in range(len(y_list)):
            sum+=(y_list[i]-ave)**2
        return sum
    def mean(nlist):
        sum=0
        for i in range(len(nlist)):
            sum+=nlist[i]
        return sum/len(nlist)
    rsqd=1-ss_res(x_list,y_list,function)/ss_tot(y_list)
    return rsqd
        
    



# def numerical_differentiator_average(x_list,f_x_list):
#     if len(x_list)!=len(f_x_list):
#         return "variable and function do not match"
#     output=[]
#     for i in range(len(x_list)):
#         if i-1==-1:
#             m1=(f_x_list[i+1]-f_x_list[i])/(x_list[i+1]-x_list[i])
#         else:
#             m1=(f_x_list[i]-(f_x_list[i-1]))/(x_list[i]-x_list[i-1])
#         if i+1==len(x_list):
#             m2=(f_x_list[i]-f_x_list[i-1])/(x_list[i]-x_list[i-1])
#         else:
#             m2=(f_x_list[i+1]-(f_x_list[i]))/(x_list[i+1]-x_list[i])
#         m=(m1+m2)/2
#         output.append(m)
#     return output

# def functionlist(x_list,f_x_list):
#     if len(x_list)!=len(f_x_list):
#         return "variable and function do not match"
#     output=[]
#     for i in range(len(x_list)):
#         print(str(x_list[i])+','+str(f_x_list[i]))
#         output.append(str(x_list[i])+','+str(f_x_list[i]))
#     return output
# def d_functionlist(x_list,f_x_list,df_x_list):
#     n=len(x_list)
#     if len(f_x_list)!=n or len(df_x_list)!=n:
#         return "variable and function and derivative do not match"
#     output=[]
#     for i in range(len(x_list)):
#         print(str(x_list[i])+','+str(f_x_list[i])+','+str(df_x_list[i]))
#         output.append(str(x_list[i])+','+str(f_x_list[i])+','+str(df_x_list[i]))
#     return output

def arange(start,end,step):
    decimals=len(str(step))-1
    steps=round((end-start)/step)
    x=start-step
    xlist=[]
    for _ in range(0,steps+1):
        x+=step
        xlist.append(round(x,decimals))
    return(xlist)

def numerical_differentiator(x,y):
    if len(x)!=len(y):
        return "variable and function do not match"
    output=[]
    for i in range(len(x)):
        if i-1==-1:
            m=(y[i+1]-y[i])/(x[i+1]-x[i])
        elif i+1==len(x):
            m=(y[i]-y[i-1])/(x[i]-x[i-1])
        else:
            m=(y[i+1]-y[i-1])/(x[i+1]-x[i-1])
        output.append(m)
    return output

def numerical_integrator(x,y,c=0):
    if len(x)!=len(y):
        return "variable and function do not match"
    output=[]
    s=0
    def mean(n1,n2):
        return (n1+n2)/2
    for i in range(len(x)):
        if i==0:
            s+=c
        elif i==len(x):
            s+=(y[i])*(x[i]-x[i-1])
        else:
            s+=mean(y[i],y[i-1])*(x[i]-x[i-1])
        output.append(s)
    return output