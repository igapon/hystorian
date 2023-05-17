from pathlib import Path

import h5py
import numpy as np

from . import utils


def convert_length_to_SI_unit(value, unit):
    unit_dic = {
        b"am": 1e-18,
        b"fm": 1e-15,
        b"pm": 1e-12,
        b"nm": 1e-9,
        b"~m": 1e-6,
        b"mm": 1e-3,
        b"cm": 1e-2,
        b"dm": 1e-1,
        b"m": 1.0,
        b"km": 1e3,
    }

    if unit not in unit_dic.keys():
        scale = 1
        print(unit)
        print("The unit type is not supported!")
    else:
        scale = unit_dic[unit]
    return value * scale


def extract_scan_info_from_header(header):
    scan_header = header.split(b"\*Ciao scan list")[1].split(b"\*")[0]
    scan_header = scan_header.strip().split(b"\r\n")
    scan_dic = {}
    for line in scan_header:
        line = line.strip(b"\\")
        # discard lines starting with an '@' for now
        if line.startswith(b"@"):
            continue
        try:
            key, value = line.split(b": ")
        except ValueError:
            continue
        if key in [b"Samps/line", b"Lines"]:
            value = int(value)
        elif key in [b"Rotate Ang.", b"Scan rate", b"Tip velocity"]:
            value = float(value)
        elif key in [b"Scan Size", b"X Offset", b"Y Offset"]:
            value, unit = value.split(b" ")
            value = convert_length_to_SI_unit(float(value), unit)

        scan_dic[key] = value
    return scan_dic


def extract_image_infos_from_header(header):
    header_of_images = header.split(b"\*Ciao image list\r\n")[1:]
    image_infos = []
    for head in header_of_images:
        head = head.strip().split(b"\r\n")
        image_infos.append(extract_image_info_from_header(head))
    return image_infos


def extract_image_info_from_header(header):
    header_dic = {}
    for line in header:
        line = line.strip(b"\\")
        # discard lines starting with an '@' for now
        if line.startswith(b"@"):
            continue
        try:
            key, value = line.split(b": ")
        except ValueError:
            continue
        if key in [b"Data offset", b"Data length", b"Bytes/pixel", b"Samps/line", b"Number of lines"]:
            value = int(value)
        elif key in [b"Scan Size"]:
            value_x, value_y, unit = value.split(b" ")
            value_x = convert_length_to_SI_unit(float(value_x), unit)
            value_y = convert_length_to_SI_unit(float(value_y), unit)
            value = np.array([value_x, value_y], dtype="float")

        header_dic[key] = value

    return header_dic


def load_nanoscope(filename):
    """
    Load a Nanoscope afm measurement files (.000, .001, etc.) and
    retrieve scan informations.

    Parameters
    ----------
    filename : string
        Filename

    Returns
    -------
    data : array-like
        image data
    scan_info : dictonary
        scan information
    image_infos : list
        image information
    header : dictonary
        information contained in the header
    """

    with open(filename, "rb") as fn:
        file_str = fn.read()

    # SEPARATE HEADER FROM DATA
    header = file_str.split(b"File list end\r\n")[0]
    # RETRIEVE SCAN PARAMETERS FROM FILE HEADER
    scan_info = extract_scan_info_from_header(header)
    image_infos = extract_image_infos_from_header(header)

    # extract the image data
    data_offset = int(image_infos[0][b"Data offset"])

    data_orig = np.frombuffer(file_str, dtype="<h", offset=data_offset)
    pixels = int(scan_info[b"Samps/line"])
    lines = int(scan_info[b"Lines"])
    data = np.zeros((len(image_infos), pixels, lines))
    start = 0
    for chan in range(len(image_infos)):
        tempX = int(image_infos[chan][b"Valid data len X"].decode("iso-8859-1"))
        tempY = int(image_infos[chan][b"Valid data len Y"].decode("iso-8859-1"))
        tempD = data_orig[start : start + tempX * tempY].reshape((tempY, tempX))
        data[chan][0:tempY, 0:tempX] = tempD
        start = start + tempX * tempY

    scan_info = {y.decode("iso-8859-1"): decode_values(scan_info, y) for y in scan_info.keys()}
    for i, k in enumerate(image_infos):
        image_infos[i] = {y.decode("iso-8859-1"): decode_values(image_infos[i], y) for y in image_infos[i].keys()}
    header = header.decode("iso-8859-1")

    header_dict = {}

    for line in header.split("\r\n"):
        if len(line.split(":")) < 2:
            continue

        header_dict[":".join(line.split(":")[:-1])[1:]] = utils.line.split(":")[-1]

    return data, scan_info, image_infos, header_dict


def decode_values(scan_dict, y):
    value = scan_dict.get(y)
    if type(value) == bytes:
        value = value.decode("iso-8859-1")
    return value


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
