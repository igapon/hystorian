import os
import numpy as np
import inspect
import struct
import h5py

DEBUG_PRINT_INFO = True
DEBUG_PRINT_WARNING = True
DEBUG_PRINT_ERROR = True


def string_scinot_to_float(string):
    split_string = string.split("E")
    float_ = float(split_string[0]) * 10 ** float(split_string[1])
    return float_


def debug_print_info(s):
    if not DEBUG_PRINT_INFO:
        return
    print("[" + str(inspect.stack()[1][3]) + "] Info: " + str(s))


def debug_print_warning(s):
    if not DEBUG_PRINT_INFO:
        return
    print("[" + str(inspect.stack()[1][3]) + "] Warning: " + str(s))


def debug_print_error(s):
    if not DEBUG_PRINT_INFO:
        return
    print("[" + str(inspect.stack()[1][3]) + "] Error: " + str(s))


def parse_header(header_data):
    header_dict = {}

    current_item = ""
    current_value = []
    for line in header_data.decode("ascii").split("\n"):
        if line.startswith(":"):
            if current_item != "":
                if len(current_value) == 1:
                    current_value = current_value[0]
                header_dict[current_item] = current_value
                current_value = []
            current_item = line
            if line == ":SCANIT_END:":
                continue
        else:
            if current_item != "":
                current_value.append(line)
    return header_dict


def parse_channel_info(header):
    if ":DATA_INFO:" not in header:
        debug_print_error("Cannot find :DATA_INFO: in header!!")
        return -1

    try:
        fullnames = header[":Scan>channels:"].split(";")
    except ValueError:
        fullnames = []

    fullnameindex = 0
    channels = []

    for c in header[":DATA_INFO:"]:
        chan = {}

        if len(c.strip()) == 0:
            continue

        c_num = c.split("\t")[1]
        c_name = c.split("\t")[2]
        c_unit = c.split("\t")[3]
        c_direction = c.split("\t")[4]
        c_calibration = c.split("\t")[5]
        c_offset = c.split("\t")[6]

        if c_num == "Channel":
            continue
        try:
            chan["Index"] = fullnames.index(c_name.replace("_", " ") + " (" + str(c_unit) + ")")
        except:
            chan["Index"] = -1

        chan["Channel"] = int(c_num)
        chan["Name"] = str(c_name)
        try:
            chan["Full Name"] = fullnames[chan["Index"]]
        except:
            chan["Full Name"] = chan["Name"]
        chan["Unit"] = str(c_unit)
        chan["Direction"] = str(c_direction)
        chan["Calibration"] = float(c_calibration)
        chan["Offset"] = float(c_offset)

        channels.append(chan)
    return channels


def get_channels(channel_info, orientation, fd, data_start_pos, pixels_x, pixels_y):
    channel_data = {}
    fd.seek(data_start_pos + 1)
    for c in channel_info:
        if c["Direction"] == "both":
            trace_data = np.reshape(
                np.array(struct.unpack(">" + str(pixels_x * pixels_y) + "f", fd.read(4 * pixels_x * pixels_y))),
                (pixels_x, pixels_y),
            )
            if orientation == "up":
                trace_data = np.flip(trace_data, axis=0)
            channel_data[c["Full Name"] + " - Trace"] = trace_data

            retrace_data = np.reshape(
                np.array(struct.unpack(">" + str(pixels_x * pixels_y) + "f", fd.read(4 * pixels_x * pixels_y))),
                (pixels_x, pixels_y),
            )
            if orientation == "up":
                retrace_data = np.flip(retrace_data, axis=0)
            channel_data[c["Full Name"] + " - Retrace"] = np.flip(retrace_data, axis=1)
        else:
            debug_print_info("Scan direction is not 'both'. Assuming single trace/retrace acquisition...")
            trace_data = np.reshape(
                np.array(struct.unpack(">" + str(pixels_x * pixels_y) + "f", fd.read(4 * pixels_x * pixels_y))),
                (pixels_x, pixels_y),
            )
            channel_data[c["Full Name"]] = trace_data
    return channel_data


def load_sxm(filename):
    header_start_token = ":NANONIS_VERSION:"
    header_end_token = ":SCANIT_END:\n\n\n"

    fd = open(filename, "rb")

    dat = fd.read(1000000)

    if dat.find(header_start_token.encode()) != 0:
        debug_print_error("Cannot find :NANONIS_VERSION: header... Is this a Nanonis file??")

    header_end_pos = dat.find(header_end_token.encode())
    data_start_pos = header_end_pos + len(header_end_token) + 1
    data_end_pos = fd.seek(0, os.SEEK_END)
    fd.seek(0)
    header_bytes = fd.read(header_end_pos + len(header_end_token) + 1)

    header = parse_header(header_bytes)
    channel_info = parse_channel_info(header)

    orientation = header[":SCAN_DIR:"]
    if ":Scan>pixels/line:" in header.keys():
        channel_data = get_channels(
            channel_info,
            orientation,
            fd,
            data_start_pos,
            int(header[":Scan>lines:"]),
            int(header[":Scan>pixels/line:"]),
        )
    else:
        px = header[":SCAN_PIXELS:"].split()[0]
        ln = header[":SCAN_PIXELS:"].split()[1]
        channel_data = get_channels(channel_info, orientation, fd, data_start_pos, int(ln), int(px))

    return [header, channel_info, channel_data]


def sxm2hdf5(filename, filepath=None):
    header, channel_info, channel_data = load_sxm(filename)

    with h5py.File(filename.split(".")[0] + ".hdf5", "w") as f:
        file_base = filepath.split(".")[0] if filepath else filename.split(".")[0]
        file_ext = filepath.split(".")[-1] if filepath else filename.split(".")[-1]

        f.create_group("metadata")
        f["metadata"].create_dataset(file_base, data=str(header))

        f.create_group("process")

        datagrp = f.create_group(f"datasets/{file_base}")
        datagrp.attrs["type"] = file_ext

        for key in channel_data:
            datagrp.create_dataset(key, data=channel_data[key])
            datagrp[key].attrs["name"] = key
            datagrp[key].attrs["shape"] = (int(header[":Scan>pixels/line:"]), int(header[":Scan>lines:"]))
            offsets = header[":SCAN_OFFSET:"].split(" ")
            while "" in offsets:
                offsets.remove("")
            datagrp[key].attrs["offset"] = (string_scinot_to_float(offsets[0]), string_scinot_to_float(offsets[1]))
            sizes = header[":SCAN_RANGE:"].split(" ")
            while "" in sizes:
                sizes.remove("")
            datagrp[key].attrs["size"] = (string_scinot_to_float(sizes[0]), string_scinot_to_float(sizes[1]))

            for info in channel_info:
                if info["Name"] == key.split(" - ")[0]:
                    for k2 in info.keys():
                        datagrp[key].attrs[k2] = info[k2]
                continue
