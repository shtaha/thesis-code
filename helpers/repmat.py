from lib.visualizer import print_matrix

import numpy as np

vect = np.random.randint(0, 2, 5)
vect = np.atleast_2d(vect).T
repmat = np.tile(vect, (1, 3))

print_matrix(vect)
print_matrix(repmat)

matrix = np.random.randint(0, 10, (5, 3))
print_matrix(matrix)

for row in matrix:
    print(row)
