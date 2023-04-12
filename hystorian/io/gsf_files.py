import numpy as np
import h5py


def gsf_read(filename):
    """Read a Gwyddion Simple Field 1.0 file format
    http://gwyddion.net/documentation/user-guide-en/gsf.html
    Args:
        filename (string): the name of the output (any extension will be replaced)
    Returns:
        metadata (dict): additional metadata to be included in the file
        data (2darray): an arbitrary sized 2D array of arbitrary numeric type
    """
    if filename.rpartition(".")[1] == ".":
        filename = filename[0 : filename.rfind(".")]

    gsfFile = open(filename + ".gsf", "rb")

    metadata = {}

    # check if header is OK
    if not (gsfFile.readline().decode("UTF-8") == "Gwyddion Simple Field 1.0\n"):
        gsfFile.close()
        raise ValueError("File has wrong header")

    term = b"00"
    # read metadata header
    while term != b"\x00":
        line_string = gsfFile.readline().decode("UTF-8")
        metadata[line_string.rpartition("=")[0]] = line_string.rpartition("=")[2]
        term = gsfFile.read(1)

        gsfFile.seek(-1, 1)

    gsfFile.read(4 - gsfFile.tell() % 4)

    # fix known metadata types from .gsf file specs
    # first the mandatory ones...
    metadata["XRes"] = np.int(metadata["XRes"])
    metadata["YRes"] = np.int(metadata["YRes"])

    # now check for the optional ones
    if "XReal" in metadata:
        metadata["XReal"] = np.float(metadata["XReal"])

    if "YReal" in metadata:
        metadata["YReal"] = np.float(metadata["YReal"])

    if "XOffset" in metadata:
        metadata["XOffset"] = np.float(metadata["XOffset"])

    if "YOffset" in metadata:
        metadata["YOffset"] = np.float(metadata["YOffset"])

    data = np.frombuffer(gsfFile.read(), dtype="float32").reshape(metadata["YRes"], metadata["XRes"])

    gsfFile.close()

    return metadata, data


def gsf2hdf5(filename, filepath=None):
    meta, data = gsf_read(filename)
    filename = "_".join(filename.split())
    filename = "".join(filename.split(".")[:-1])
    filename = filename.replace(".", "-") + ".gsf"

    with h5py.File(filename.split(".")[0] + ".hdf5", "w") as f:
        name = filename.split(".")[0:-1]
        name = "_".join(name)

        file_base = filepath.split(".")[0] if filepath else filename.split(".")[0]
        file_ext = filepath.split(".")[-1] if filepath else filename.split(".")[-1]

        f.create_group("metadata")
        f["metadata"].create_dataset(file_base, data=str(meta))

        f.create_group("process")

        datagrp = f.create_group(f"datasets/{file_base}")
        datagrp.attrs["type"] = file_ext

        datagrp.create_dataset(name, data=data)

        datagrp[name].attrs["name"] = name
        datagrp[name].attrs["shape"] = data.shape
        datagrp[name].attrs["size"] = (meta["XRes"], meta["YRes"])
        datagrp[name].attrs["offset"] = (meta["XOffset"], meta["YOffset"])
        datagrp[name].attrs["WavenumberScaling"] = float(meta["Neaspec_WavenumberScaling"])
