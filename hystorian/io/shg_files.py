import numpy as np
import h5py


def dat2hdf5(filename, filepath=None, params=None):
    data = np.fromfile(filename, sep=" ")
    try:
        data = np.reshape(data, (int(np.sqrt(len(data))), int(np.sqrt(len(data)))))
    except ValueError:
        raise ValueError("Shape of the data is not a square, keep it 1D")

    if params is not None:
        contents = np.fromfile(params, sep=" ")
        try:
            contents = np.reshape(contents, (3, int(len(contents) / 3)))
        except ValueError:
            raise ValueError("Shape is not 3xN, keep it 1D")

    with h5py.File(filename.split(".")[0] + ".hdf5", "w") as f:
        file_base = filepath.split(".")[0] if filepath else filename.split(".")[0]
        file_ext = filepath.split(".")[-1] if filepath else filename.split(".")[-1]

        f.create_group("metadata")
        if params is not None:
            f["metadata"].create_dataset(file_base, data=contents)

        f.create_group("process")

        datagrp = f.create_group(f"datasets/{filepath.split('.')[0]}")
        datagrp.attrs["type"] = file_ext

        datagrp.create_dataset("SHG", data=data)
