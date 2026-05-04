import numpy as np
from pprint import pprint

from example import load_complex_zip
from Question_A1 import build_complex

def boundary_matrix(complex, k):
    # handle edge cases
    if k == 0:
        n_cols = len(complex[0]) if len(complex) > 0 else 0
        return np.zeros((1, n_cols)).T
    if k >= len(complex):
        return np.zeros((0, 0), dtype=int)
    # extract relevant simplices from complex data structure
    lower_simplices = complex[k - 2] if k>1 else {(): 0}
    higher_simplices = complex[k - 1]

    # initialise d_k as a matrix of zeros
    d_matrix = np.zeros((len(lower_simplices), len(higher_simplices)))

    # iterate through each k-dimensional simplex
    for simplex, col in higher_simplices.items():
        simplex = tuple(simplex)

        # iterate through each face of the simplex
        for i in range(len(simplex)):
            # remove i_th component of list
            face = simplex[:i] + simplex[i + 1:]
            row = lower_simplices[face]
            # use formula for d_k
            d_matrix[row, col] = (-1) ** i

    return d_matrix


if __name__ == "__main__":
    print("triangle.npz:")
    data = load_complex_zip("complexes.zip", "triangle.npz")
    complex = build_complex(data)
    for i in range(0, 3):
        print(f"Printing boundary operator for k={i}")
        matrix = boundary_matrix(complex, i)
        print(matrix)

    print("triangle_minus_triangle.npz:")
    data = load_complex_zip("complexes.zip", "triangle_minus_triangle.npz")
    complex = build_complex(data)
    for i in range(0, 3):
        print(f"Printing boundary operator for k={i}")
        matrix = boundary_matrix(complex, i)
        print(matrix)