# Candidate Number: 1104704


from itertools import combinations
from pprint import pprint
import numpy as np
import zipfile
from numpy.linalg import matrix_rank as rank


def load_complex_npz(file):
    data = np.load(file)
    cells = data["cells"]
    coordinates = data["coordinates"]
    return (cells, coordinates)

# Load a complex from within the .zip
def load_complex_zip(zipname, complexname):
    with zipfile.ZipFile(zipname) as z:
        with z.open(complexname) as f:
            return load_complex_npz(f)

# example code:
# connections between highest dimensional cells and vertices
cells = [[0, 1, 2]]
# coordinates of each vertex
coordinates = [[0, 0], [1, 0], [0.5, 0.5]]

# visualise complex
from plot_complex import plot_complex

plot_complex(cells, coordinates)

import zipfile
# Print out files inside the .zip
with zipfile.ZipFile("complexes.zip") as z:
    filenames = z.namelist()
print(f"Available filenames: {filenames}")


print("Question A.1")


def build_complex(data):
    # split data into topology and geometry
    cells, coordinates = data
    cells = cells.tolist()
    # initialise the "complex" data structure (a list of dictionaries)
    complex = []
    # iterate through cells
    for cell in cells:
        # iterate through dimensions of current cell
        for dim in range(1,len(cell)+1):
            # assure complex has enough dictionaries for each dimension
            if len(complex) < dim:
                complex.append({})
            # get combinations of length corresponding to current dimension
            tuples = list(combinations(cell, r=dim))
            # dimension 1 must be identity
            if dim == 1:
                for tuple in tuples:
                    complex[dim-1][tuple] = tuple[0]
                continue
            if complex[dim-1]:
                biggest_id = max(complex[dim-1].values())+1
            else:
                biggest_id = 0
            for tuple in tuples:
                if tuple not in complex[dim-1].keys():
                    complex[dim-1][tuple] = biggest_id
                    biggest_id += 1
    # sort each dictionary in complex by value
    sorted_complex = [{k: v for k,v in sorted(dict.items(), key=lambda item: item[1])} for dict in complex]
    return sorted_complex

# task 1
# triangle.npz
print("triangle.npz:")
data = load_complex_zip("complexes.zip", "triangle.npz")

complex = build_complex(data)

for (dim, c) in enumerate(complex):
    print(f"Printing complex of dimension {dim}")
    pprint(c, sort_dicts=False)

print("\n\n")

# triangle_minus_triangle.npz
data = load_complex_zip("complexes.zip", "triangle_minus_triangle.npz")

print("triangle_minus_triangle.npz:")

print(data)

complex = build_complex(data)

for (dim, c) in enumerate(complex):
    print(f"Printing complex of dimension {dim}")
    pprint(c, sort_dicts=False)

print("\n\n")


print("Question A.2")


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


print("Question A.3")


def build_boundary_operators(complex):
    for i in range(0, len(complex) + 1):
        print(f"Printing boundary operator for k={i}")
        matrix = boundary_matrix(complex, i)
        print(matrix)

def test_boundary_of_boundary(complex):
    print(boundary_matrix(complex, 0))
    print(boundary_matrix(complex, 1))
    assert all(
        [
            # ensure matrix contains only zeros
            not np.any(
            # matrix multiply d_k and d_k+1
            boundary_matrix(complex, i) @ boundary_matrix(complex, i + 1))
            # loop through all valid dimensions
            for i in range(0, len(complex)-1)]
    )
    print("test for A.1.11 passed")


with zipfile.ZipFile("complexes.zip") as z:
    filenames = z.namelist()

for f in filenames:
    print(f"{f}:")
    data = load_complex_zip("complexes.zip", "triangle_minus_triangle.npz")
    complex = build_complex(data)
    build_boundary_operators(complex)
    test_boundary_of_boundary(complex)


print("Question A.4")


# starting with (A.2.8):
# b_k = dim(Z_k) - dim(B_k) = dim ker(d_k) - dim im(d_k+1)
# let D_k be the matrix of the boundary operator:
# d_k : C_k(K) -> C_k-1(K)
# then:
# dim im(d_k) = rank(D_k)
# so:
# dim B_k = dim im(d_k+1) = rank(D_k+1)
# using the rank-nullity theorem on d_k:
# dim C_k dim ker(d_k) + rank(D_k)
# and rearranging:
# dim ker(d_k) = dim C_k - rank(D_k)
# but dim C_k is just ```len(complex[k])```
# therefore:
# dim Z_k = dim ker(d_k) = dim C_k - rank(D_k)
# and finally substituting this into the Betti number formula:
# b_k = dim Z_k - dim B_k
# gives you:
# b_k = dim C_k - rank(d_k) - rank(d_k+1)


print("Question A.5")


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


with zipfile.ZipFile("complexes.zip") as z:
    filenames = z.namelist()

for f in filenames:
    print(f"{f}:")
    data = load_complex_zip("complexes.zip", f)
    b = betti_numbers(data)
    print(f"Betti numbers: {b}")
