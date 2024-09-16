'''
# Vercil A. Juan

Applied Physics 155
III - BS Physics
2022-12948
'''
import random

# Helper functions
# Creates a matrix of 0's with a given row and column (list version of np.zeroes)
def zero_matrix(rows,cols):
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    return output

# Outputs the size of a matrix (list version of np.size)
def mat_size(matrix):
    # check if the matrix has same length rows
    rowlen = len(matrix[0])
    for row in matrix:
        if len(row) != rowlen:
            raise ValueError('Matrix given is not actually a matrix (unequal row length)')
    else:
        return len(matrix), len(matrix[0])
    
# A.1
# Creates a matrix of random integers with rows and columns

def random_integer_matrix(nrows,ncols):
    output = [[random.randint(1,9) for _ in range(ncols)] for _ in range(nrows)]
    return output

# A.2
# Creates a dot product of two vectors

def scalar_product(list1,list2):
    length = len(list1)
    if len(list2)!=length:
        raise ValueError('Vector length not equal, cannot take dot product')
    output = 0
    for i in range(length):
        output += list1[i] * list2[i]
    return output

# A.3
# Transposes a given matrix

def transpose_matrix(orig_matrix):
    row,col = mat_size(orig_matrix)
    output_matrix = zero_matrix(col,row)
    for i in range(row):
        for j in range(col):
            output_matrix[j][i] = orig_matrix[i][j]
    return output_matrix
    
# A.4
# Multiplies two given matrices

def matrix_multiply(matrix1, matrix2):
    r1, c1 = mat_size(matrix1)
    r2, c2 = mat_size(matrix2)
    if c1 != r2:
        raise ValueError('Matrices do not match, cannot get matrix product')
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
rows,cols = 4,5

print('Random integer matrix :', random_integer_matrix(rows,cols))
print('Scalar product :', scalar_product(vector1,vector2))
print('Transpose matrix :' , transpose_matrix(matrix3))
print('Matrix multiplication :' , matrix_multiply(matrix1,matrix2))


# Accuracy testing
# Importing numpy to verify correct answers

import numpy as np

matrix1 = np.array(matrix1)
matrix2 = np.array(matrix2)
matrix3 = np.array(matrix3)
vector1 = np.array(vector1)
vector2 = np.array(vector2)

print('Correct randint matrix :\n',np.random.randint(1, 10, size=(rows, cols)))
print('Correct scalar product :' , np.dot(vector1,vector2))
print('Correct transpose matrix :\n' , np.transpose(matrix3))
print('Correct matrix multiply :\n', matrix1 @ matrix2)