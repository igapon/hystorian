import csv
import pathlib

import h5py
import numpy as np


def extract_csv(filename, delimiter=","):
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter)
        data = list(data)
        header = data[0]
        data = data[1:]
        np_data = np.array(data).astype("S")

    return data, header


def csv2hdf5(filename, filepath=None):
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter=",")
        data = list(data)
        header = data[0]
        data = data[1:]
        np_data = np.array(data).astype("S")

    file_base = filepath.split(".")[0] if filepath else filename.split(".")[0]
    file_ext = filepath.split(".")[-1] if filepath else filename.split(".")[-1]

    with h5py.File(f"{filename.split('.')[0]}.hdf5", "w") as f:
        f.create_group("metadata")
        f["metadata"].create_dataset(file_base, data=str(header))

        f.create_group("process")

        datagrp = f.create_group(f"datasets/{file_base}")
        datagrp.attrs["type"] = file_ext

        dataset = datagrp.create_dataset(file_base, data=np_data)
        dataset.attrs["name"] = file_base
        dataset.attrs["shape"] = np_data.shape
        dataset.attrs["header"] = header
