from typing import List, Dict, Tuple
from itertools import combinations
from pprint import pprint

from example import load_complex_zip

"""
task: enumerate all the simplices from the compressed representation 'cells'
"""



""""""
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

if __name__ == "__main__":
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