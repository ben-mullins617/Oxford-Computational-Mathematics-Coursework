import zipfile
import numpy as np

from example import load_complex_zip
from Question_A1 import build_complex
from Question_A2 import boundary_matrix

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


if __name__ == "__main__":
    with zipfile.ZipFile("complexes.zip") as z:
        filenames = z.namelist()

    for f in filenames:
        print(f"{f}:")
        data = load_complex_zip("complexes.zip", "triangle_minus_triangle.npz")
        complex = build_complex(data)
        build_boundary_operators(complex)
        test_boundary_of_boundary(complex)