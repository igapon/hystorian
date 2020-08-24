import numpy as np
import h5py


def dat2hdf5(filename, params=None):
    data = np.fromfile(filename,sep=" ")
    try:
        data = np.reshape(data, (int(np.sqrt(len(data))), int(np.sqrt(len(data)))))
    except:
        print("Shape of the data is not a square, keep it 1D")

    if params is not None:
        contents = np.fromfile(params, sep=" ")
        try:
            contents = np.reshape(contents, (3, int(len(contents)/3)))
        except:
            print("Shape is not 3xN, keep it 1D")

    with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
        typegrp = f.create_group("type")
        typegrp.create_dataset(filename.split('.')[0], data=filename.split('.')[-1])

        metadatagrp = f.create_group("metadata")
        if params is not None:
            metadatagrp.create_dataset(filename.split('.')[0], data=contents)

        f.create_group("process")

        datagrp = f.create_group("datasets/" + filename.split('.')[0])
        datagrp.create_dataset("SHG", data=data)