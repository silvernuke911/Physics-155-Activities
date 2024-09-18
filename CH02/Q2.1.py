import numpy as np 
import matplotlib.pyplot as plt

def quadratic_formula(coeff_list):
    a,b,c = coeff_list[0], coeff_list[1], coeff_list[2]
    discriminant = b**2 - 4 * a * c
    if discriminant < 0 :
        return 'Zeros does not exist'
    else:
        x1 = (-b - np.sqrt(discriminant)) / (2*a)
        x2 = (-b + np.sqrt(discriminant)) / (2*a)
        return x1,x2
print(quadratic_formula([1,9,1]))

