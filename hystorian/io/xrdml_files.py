import h5py
import re
import numpy as np


# ==========================================
# XRDML conversion


def xrdml2hdf5(filename, filepath=None):
    with open(filename, "r") as f:
        contents = f.read()

    with h5py.File(f"{filename.split('.')[0]}.hdf5", "w") as f:
        param_names = ["counts", "2theta", "omega", "phi", "chi", "x", "y"]
        param_headers = [
            "",
            '<positions axis="2Theta" unit="deg">',
            '<positions axis="Omega" unit="deg">',
            '<positions axis="Phi" unit="deg">',
            '<positions axis="Chi" unit="deg">',
            '<positions axis="X" unit="mm">',
            '<positions axis="Z" unit="mm">',
        ]

        scans = contents.split("<dataPoints>")[1:]

        dataset_name = filepath.split(".")[0] if filepath else filename.split(".")[0]
        extension_name = filepath.split(".")[-1] if filepath else filename.split(".")[1]

        f.create_group("metadata")
        f["metadata"].create_dataset(dataset_name, data=contents)

        f.create_group("process")

        datagrp = f.create_group(f"datasets/{dataset_name}")
        datagrp.attrs["type"] = extension_name

        for i, name in enumerate(param_names):
            all_data = []
            for scan in scans:
                if i == 0:
                    counts = scan.split('"counts">')[-1].split("</")[0].split()
                    val_list = list(map(float, counts))
                    step_num = len(val_list)
                elif i != 0:
                    data_range = scan.split(param_headers[i])[1].split("</positions>")[0]
                    if len(data_range.split()) == 1:
                        val = float(re.findall(r"[-+]?\d*\.\d+|\d+", data_range.split()[0])[0])
                        val_list = np.full(step_num, val)
                    elif len(data_range.split()) == 2:
                        first_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", data_range.split()[0])[0])
                        last_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", data_range.split()[1])[0])
                        val_list = np.arange(first_val, last_val, (last_val - first_val) / step_num)
                all_data.append(val_list)

            all_data = np.array(all_data).T
            datagrp.create_dataset(name, data=all_data)
            datagrp[name].attrs["shape"] = np.shape(all_data)
            datagrp[name].attrs["name"] = "counts"
            datagrp[name].attrs["size"] = 0
            datagrp[name].attrs["offset"] = 0
            datagrp[name].attrs["unit"] = "au"

            if name == "x" or name == "y" or name == "z":
                datagrp[name].attrs["unit"] = "mm"
            elif name == "counts":
                datagrp[name].attrs["unit"] = "au"
            else:
                datagrp[name].attrs["unit"] = "deg"
