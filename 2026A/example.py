import numpy as np
import zipfile

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

if __name__ == "__main__":
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