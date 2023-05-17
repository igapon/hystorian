from pathlib import Path

import h5py
import numpy as np

from . import utils


def load_nanoscope(path):
    with open(path, "rb") as fn:
        file_str = fn.read().decode("iso-8859-1")

        header = file_str.split("File list end\r\n")[0]
        scan_info = extract_scan_info(header)

        header_of_images = header.split("\\*Ciao image list\r\n")[1:]
        image_infos = [extract_image_info(head.split("\r\n")) for head in header_of_images]

        data_offset = int(image_infos[0]["Data offset"])
        with open(path, "rb") as fn:
            data_orig = np.frombuffer(fn.read(), dtype="<h", offset=data_offset)
        pixels = int(scan_info["Samps/line"])
        lines = int(scan_info["Lines"])
        data = {}  # np.zeros((len(image_infos), pixels, lines))
        start = 0
        for chan in range(len(image_infos)):
            tempX = int(image_infos[chan]["Valid data len X"])
            tempY = int(image_infos[chan]["Valid data len X"])
            tempD = data_orig[start : start + tempX * tempY].reshape((tempY, tempX))
            data[image_infos[chan]["@2:Image Data"].split('"')[1]] = tempD
            start = start + tempX * tempY

        return data, scan_info, image_infos


def extract_scan_info(header):
    scan_header = header.split("\\*Ciao scan list")[1].split("\\*")[0]
    scan_dict = {}

    for line in scan_header.split("\r\n"):
        if len(line.split(":")) < 2:
            continue
        key, val = ":".join(line.split(":")[:-1])[1:], line.split(":")[-1].strip()

        scan_dict[key] = utils.conversion_metadata(conversion_units(val))
    return scan_dict


def extract_image_info(header):
    header_dict = {}

    for line in header:
        line = line.strip("\\")
        if len(line.split(":")) < 2:
            continue

        key = ":".join(line.split(":")[:-1])
        value = line.split(":")[-1]

        header_dict[key] = utils.conversion_metadata(conversion_units(value))

    return header_dict


def conversion_units(dat):
    unit_dic = {
        "am": 1e-18,
        "fm": 1e-15,
        "pm": 1e-12,
        "nm": 1e-9,
        "~m": 1e-6,
        "mm": 1e-3,
        "cm": 1e-2,
        "dm": 1e-1,
        "m": 1.0,
        "km": 1e3,
    }
    if len(dat.split(" ")) == 2:
        value, unit = dat.split(" ")
        if unit in unit_dic and utils.is_number(value):
            return float(value) * unit_dic[unit]

    return dat


"""
def nanoscope2hdf5(filename, filepath=None):
    data, scan_info, image_infos, header = load_nanoscope(filename)
    with h5py.File(filename.replace(".", "_") + ".hdf5", "w") as f:
        metadatagrp = f.create_group("metadata")
        f.create_group("process")

        if filepath is not None:
            metadatagrp.create_dataset(filepath.replace(".", "_"), data=header)
            datagrp = f.create_group("datasets/" + filepath.replace(".", "_"))
        else:
            metadatagrp.create_dataset(filename.replace(".", "_"), data=header)
            datagrp = f.create_group("datasets/" + filename.replace(".", "_"))

        datagrp.attrs.__setattr__("type", "Nanoscope")

        for key in scan_info.keys():
            datagrp.attrs[key] = scan_info[key]
        # Get the name and trace orientation of each the channels
        nameChan = []
        for i in range(1, len(header.split("\@2:Image Data: "))):
            chan = header.split("\@2:Image Data: ")[i].split("\r\n")[0].split()[-1][1:-1]
            trace = header.split("\Line Direction: ")[i].split("\r\n")[0].split()[-1]
            full_name = chan + "_" + trace
            nameChan.append(full_name)

        for indx, name in enumerate(nameChan):
            dtst = datagrp.create_dataset(name, data=data[indx])
            for key in image_infos[indx].keys():
                dtst.attrs[key] = image_infos[indx][key]
            dtst.attrs["scale_m_per_px"] = scan_info["Scan Size"] / scan_info["Samps/line"]
"""
