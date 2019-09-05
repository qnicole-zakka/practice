# review basic Python

import numpy as np
import os, sys

# Euclidean distance
def l2(x, y):
    return np.sqrt(np.sum((x - y) ** 2))    
    

# distance longer than 5
def gt5(x):
    bool_mask = (x > 5)
    return x[bool_mask]

# Multiplication
def mulCompare(x, y):
    print('x = {}\ny = {}\n'.format(x, y))
    print('* or np.multiply is elementwise product:\n {}\n'.format(x * y))
    print('x.dot(y) or np.dot(x, y) is matrix multiply:\n {}\n'.format(np.dot(x, y)))


def arrayCompute(x):
    print('x = {}\ntranspose of x = {}\n'.format(x, x.T))
    print('sum of all elements: {}'.format(np.sum(x)))
    print('sum of each column: {}'.format(np.sum(x, axis=0)))
    print('sum of each row: {}'.format(np.sum(x, axis=1)))

# broadcasting
def broadcast(x, y):
    print('Broadcasting:\nshape of x {}  = {}, shape of y {} = {}'.format(x, x.shape, y, y.shape))
    print('x + y = {}'.format(x + y))
    # Compute outer product of vectors
    v = np.array([1,2,3])  # v has shape (3,)
    w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
    print(np.reshape(v, (3, 1)) * w)

# Add a vector to each row of a matrix
    x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
    print(x + v)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
    print((x.T + w).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
    print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
    print(x * 2)                


def main():
    print('Current running script is: {}\n \
            running at {}'.format(__file__, os.path.abspath(__file__)))
    x = np.array([0, 3, 5])
    y = np.array([4, 5, 5])
    print(l2(x, y))
    
    z = np.array([[1, 3, 5], [7, 9, 11]])
    print('original vector is {}, after gt5 is {}\n'.format(z, gt5(z)))

    x1 = np.array([[1, 2], [3, 4]])
    y1 = np.array([[5, 6], [7, 8]])
    mulCompare(x1, y1)
    print('z = {} and array computation results: \n'.format(z))
    arrayCompute(z)

    y2 = np.array([1, 0])
    broadcast(x1, y2)
    x2 = np.array([[1, 2], [3, 4], [5, 6]])
    y3 = np.array([[1], [0], [1]])
    broadcast(x2, y3)
    
if __name__ == '__main__':
    main()
