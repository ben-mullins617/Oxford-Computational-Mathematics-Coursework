import numpy as np
import pyvista as pv

def plot_complex(cells, coordinates, *, force_dim=None, **kwargs):
    """
    Dispatch to the 2D or 3D plotting routine depending on the coordinates.

    Parameters
    ----------
    cells, coordinates : as for plot_2d_complex / plot_3d_complex
    force_dim : {None, 2, 3}
        If set, forces dispatch to 2D or 3D regardless of coordinates shape.
    **kwargs : passed to the selected plotting function.
    """
    import numpy as _np

    pts = _np.asarray(coordinates)
    if pts.ndim != 2:
        raise ValueError("coordinates must be an (N,2) or (N,3) array-like")

    if force_dim is not None:
        if force_dim == 2:
            return plot_2d_complex(cells, coordinates, **kwargs)
        elif force_dim == 3:
            return plot_3d_complex(cells, coordinates, **kwargs)
        else:
            raise ValueError("force_dim must be None, 2, or 3")

    if pts.shape[1] == 2:
        return plot_2d_complex(cells, coordinates, **kwargs)
    elif pts.shape[1] == 3:
        # if all z are zero, treat as 2D by default
        if _np.allclose(pts[:, 2], 0.0):
            return plot_2d_complex(cells, coordinates, **kwargs)
        return plot_3d_complex(cells, coordinates, **kwargs)
    else:
        raise ValueError("coordinates must have 2 or 3 columns")


def plot_2d_complex(cells, coordinates, *,
                 color="#4C72B0",
                 edge_color="black",
                 show_edges=True,
                 smooth_shading=False,
                 background="white",
                 window_size=(700,500)):
    """
    Plot a planar complex given 2D (or 3D) coordinates and cells (list/array of index lists).
    - coordinates: (N,2) or (N,3) array-like
    - cells: (M, k) array-like with k==3 (triangles) or k==4 (quads)
    """
    pts = np.asarray(coordinates)
    cells_arr = np.asarray(cells, dtype=np.int64)

    # --- validate coordinates ---
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError("coordinates must be shape (N,2) or (N,3)")

    # --- convert to 3D if needed ---
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=float)])

    # --- validate cells ---
    if cells_arr.ndim != 2 or cells_arr.shape[1] not in (3, 4):
        raise ValueError("cells must be an (M,3) triangle array or (M,4) quad array")
    if cells_arr.max() >= len(pts) or cells_arr.min() < 0:
        raise IndexError("cell indices out of range for coordinates array")

    # --- build VTK face buffer: [n, v0, v1, v2, n, v0, ...] ---
    nverts = cells_arr.shape[1]
    prefix = np.full((cells_arr.shape[0], 1), nverts, dtype=np.int64)
    faces_vtk = np.hstack([prefix, cells_arr]).ravel()

    # --- construct PyVista PolyData and plot ---
    mesh = pv.PolyData(pts, faces_vtk)

    p = pv.Plotter(window_size=window_size)
    p.set_background(background)

    p.add_mesh(
        mesh,
        color=color,
        show_edges=show_edges,
        edge_color=edge_color,
        smooth_shading=smooth_shading,
    )

    # flat top-down view for 2D meshes
    p.camera_position = "xy"
    p.show()


def plot_3d_complex(cells, coordinates, *,
                 cell_type=None,            # "auto" (default), "tris", "tets"
                 color="#4C72B0",
                 edge_color="white",
                 show_edges=True,
                 smooth_shading=True,
                 opacity=1.0,
                 show_wireframe=True,
                 extract_surface_for_shading=True,
                 background="white",
                 window_size=(900, 700),
                 camera="iso"):
    """
    Plot a 3D complex given 3D coordinates and cells.

    - coordinates: (N,3) array-like
    - cells:  (M,k) array-like, where k==3 => triangles (surface)
                                  k==4 => tets (volume)
    - cell_type: override autodetection: "tris" or "tets" or None/"auto"
    """
    pts = np.asarray(coordinates, dtype=float)
    cells_arr = np.asarray(cells, dtype=np.int64)

    # --- validate coordinates ---
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("coordinates must be shape (N,3) for 3D plotting")

    # --- determine cell type ---
    if cell_type is None or cell_type == "auto":
        if cells_arr.ndim != 2:
            raise ValueError("cells must be a 2D array of indices")
        if cells_arr.shape[1] == 3:
            kind = "tris"
        elif cells_arr.shape[1] == 4:
            kind = "tets"
        else:
            raise ValueError("cells must be (M,3) triangles or (M,4) tetrahedra")
    else:
        kind = cell_type.lower()
        if kind not in ("tris", "tets"):
            raise ValueError('cell_type must be "tris", "tets", or None')

    # --- index check ---
    if cells_arr.size > 0:
        if cells_arr.min() < 0 or cells_arr.max() >= len(pts):
            raise IndexError("cell indices out of range for coordinates array")

    # --- build mesh & plot ---
    p = pv.Plotter(window_size=window_size)
    p.set_background(background)

    mesh = None

    if kind == "tris":
        # PyVista expects face buffer like [3, v0, v1, v2, 3, v0, ...]
        prefix = np.full((cells_arr.shape[0], 1), 3, dtype=np.int64)
        faces_vtk = np.hstack([prefix, cells_arr]).ravel()
        mesh = pv.PolyData(pts, faces_vtk)

        p.add_mesh(
            mesh,
            color=color,
            show_edges=show_edges,
            edge_color=edge_color,
            smooth_shading=smooth_shading,
            opacity=opacity,
        )

    else:  # tets
        # cells_vtk: [4, v0, v1, v2, v3, 4, ...] flattened
        prefix = np.full((cells_arr.shape[0], 1), 4, dtype=np.int64)
        cells_vtk = np.hstack([prefix, cells_arr]).ravel()

        # one celltype per cell
        celltypes = np.full(cells_arr.shape[0], pv.CellType.TETRA, dtype=np.uint8)

        mesh = pv.UnstructuredGrid(cells_vtk, celltypes, pts)

        if extract_surface_for_shading:
            surface = mesh.extract_surface(algorithm="dataset_surface")
            surface.compute_normals(inplace=True)
            p.add_mesh(
                surface,
                color=color,
                show_edges=show_edges,
                edge_color=edge_color,
                smooth_shading=smooth_shading,
                opacity=opacity,
            )
        else:
            # draw volumetric tets directly (usually looks better to show surface)
            p.add_mesh(
                mesh,
                color=color,
                show_edges=show_edges,
                edge_color=edge_color,
                smooth_shading=smooth_shading,
                opacity=opacity,
            )

        if show_wireframe:
            p.add_mesh(mesh, style="wireframe", color=[0.35, 0.35, 0.35], opacity=0.2)

    # camera
    if camera is not None:
        p.camera_position = camera

    p.show()

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Failed to use notebook backend.*",
    category=UserWarning,
)

# --- Example usage: single triangle (2D coords) ------------------------------
if __name__ == "__main__":
    coordinates2d = np.array([[0.0, 0.0],
                         [1.0, 0.0],
                         [0.5, 0.8]])       # (3,2)

    tri_cells = np.array([[0, 1, 2]])      # (1,3)

    plot_complex(tri_cells, coordinates2d)

    points3d = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ])

    tets = np.array([[0, 1, 2, 3]])
    plot_complex(tets, points3d)
