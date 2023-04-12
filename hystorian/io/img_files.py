import numpy as np
import h5py
from PIL import Image


def image2hdf5(filename, filepath=None):
    img = Image.open(filename)
    arr = np.array(img)

    with h5py.File(filename.split(".")[0] + ".hdf5", "w") as f:
        file_base = filepath.split(".")[0] if filepath else filename.split(".")[0]
        file_ext = filepath.split(".")[-1] if filepath else filename.split(".")[-1]

        f.create_group("metadata")
        f["metadata"].create_dataset(file_base, data="None")

        f.create_group("process")

        datagrp = f.create_group(f"datasets/{file_base}" + filepath.split(".")[0])
        datagrp.attrs["type"] = file_ext

        keys = ["red", "green", "blue"]
        for indx, key in enumerate(keys):
            datagrp.create_dataset(key, data=arr[:, :, indx])
            datagrp[key].attrs["name"] = key + " channel"
            datagrp[key].attrs["shape"] = arr[:, :, indx].shape
            datagrp[key].attrs["size"] = (len(arr[:, :, indx]), len(arr[:, :, indx][0]))
            datagrp[key].attrs["offset"] = (0, 0)
