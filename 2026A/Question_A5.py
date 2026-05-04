import zipfile
import numpy as np
from numpy.linalg import matrix_rank as rank

from example import load_complex_zip
from Question_A1 import build_complex
from Question_A2 import boundary_matrix

def betti_numbers(cells):
    complex = build_complex(cells)
    b = [1]
    # calculate Betti number for each dimension
    for k in range(0,2):
        # using the formula from A.4:
        dim_C_k = len(complex[k])
        rank_d_K = rank(boundary_matrix(complex, k))
        rank_d_K_plus_1 = rank(boundary_matrix(complex, k+1))
        b_k = dim_C_k - rank_d_K - rank_d_K_plus_1
        # cast from numpy int to int, append to list of Betti numbers
        b.append(int(b_k))
    return b

if __name__ == '__main__':
    with zipfile.ZipFile("complexes.zip") as z:
        filenames = z.namelist()

    for f in filenames:
        print(f"{f}:")
        data = load_complex_zip("complexes.zip", f)
        b = betti_numbers(data)
        print(f"Betti numbers: {b}")