'''
Vercil A. Juan
Physics 155.1
III - BS Physics

2022-12948
'''
import random

# Helper function
# Creates a matrix of 0's with a given row and column (list version of np.zeroes)

def zero_matrix(rows,cols):
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    return output

# A1
# Creates a matrix of random integers with rows and columns

def random_integer_matrix(nrows,ncols):
    outputlist = []
    for _ in range(nrows):
        outputlist.append([random.randint(1,9) for _ in range(ncols)])
    return outputlist

# A2
# Creates a dot product of two vectors

def scalar_product(list1,list2):
    length = len(list1)
    if len(list2)!=length:
        raise ValueError('Vector length not equal, cannot take dot product')
    output = 0
    for i in range(length):
        output += list1[i] * list2[i]
    return output

# A3
# Transposes a given matrix

def transpose_matrix(orig_matrix):
    row,col = len(orig_matrix), len(orig_matrix[0])
    output_matrix = zero_matrix(col,row)
    for i in range(row):
        for j in range(col):
            output_matrix[j][i] = orig_matrix[i][j]
    return output_matrix
    
# A4
# Multiplies two given matrices

def mat_multi(matrix1, matrix2):
    r1, c1 = len(matrix1), len(matrix1[0])
    r2, c2 = len(matrix2), len(matrix2[0])
    if c1 != r2:
        return 'Matrices do not match, cannot get product'
    else:
        output = zero_matrix(r1, c2)
        tempmat = transpose_matrix(matrix2)
        for i in range(r1):
            for j in range(c2):
                row = matrix1[i]
                col = tempmat[j]
                output[i][j] = scalar_product(row, col)
    return output

matrix1 = [[1,2],[3,4]]
matrix2 = [[5,6],[7,8]]
matrix3 = [[1,2,3],[4,5,6]]

vector1 = [1,2]
vector2 = [3,4]

print('Random integer matrix :', random_integer_matrix(5,2))
print('Scalar product :', scalar_product(vector1,vector2))
print('Transpose matrix :' , transpose_matrix(matrix3))
print('Matrix multiplication :' , mat_multi(matrix1,matrix2))

# Accuracy testing
# Importing numpy to verify code answers
import numpy as np
matrix1 = np.array(matrix1)
matrix2 = np.array(matrix2)
print(np.dot(vector1,vector2))
print(np.transpose(matrix3))
print(matrix1@matrix2)




# def mat_mult(matrix1, matrix2):
#     r1, c1 = len(matrix1), len(matrix1[0])
#     r2, c2 = len(matrix2), len(matrix2[0])
#     if c1 != r2:
#         return 'Matrices do not match, cannot get product'
#     else:
#         output = zero_matrix(r1,c2)
#         for i in range(r1):
#             for j in range(c2):
#                 for k in range(c1):
#                     output[i][j] += matrix1[i][k] * matrix2[k][j]
#     return output
# print(mat_mult(matrix1,matrix2))