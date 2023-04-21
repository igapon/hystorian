from igor2 import binarywave
import h5py
import numpy as np
import pathlib

# ==========================================
# IBW conversion


def extract_ibw(filename):
    def correct_label(label):
        label = [x for x in label if x]  # Remove the empty lists
        label = label[0]  # Remove the unnecessary inception

        # Correct the duplicate letters
        label = [i.decode("UTF-8") for i in label if len(i) > 0]
        label = [x.split("Trace")[0] + "Trace" if "Trace" in x else x for x in label]
        label = [x.split("Retrace")[0] + "Retrace" if "Retrace" in x else x for x in label]
        label = [x.encode() for x in label if len(x) > 0]

        return label

    if isinstance(filename, str):
        filename = pathlib.Path(filename)

    tmpdata = binarywave.load(filename)["wave"]

    metadata = tmpdata["note"]

    label_list = correct_label(tmpdata["labels"])

    data = {}
    attributes = {}

    fastsize = float(str(metadata).split("FastScanSize:")[-1].split("\\r")[0])
    slowsize = float(str(metadata).split("SlowScanSize:")[-1].split("\\r")[0])
    xoffset = float(str(metadata).split("XOffset:")[1].split("\\r")[0])
    yoffset = float(str(metadata).split("YOffset:")[1].split("\\r")[0])

    for i, k in enumerate(label_list):
        if len(np.shape(tmpdata["wData"])) == 2:
            data[k] = np.flipud(tmpdata["wData"][:, i].T)
            shape = tmpdata["wData"][:, i].T.shape
        else:
            data[k] = np.flipud(tmpdata["wData"][:, :, i].T)
            shape = tmpdata["wData"][:, :, i].T.shape

        attributes[k] = {}

        attributes[k]["shape"] = shape
        attributes[k]["scale_m_per_px"] = fastsize / shape[0]
        attributes[k]["name"] = k.decode("utf8")
        attributes[k]["size"] = (fastsize, slowsize)
        attributes[k]["offset"] = (xoffset, yoffset)
        # attributes[k]["path"] = f"datasets/{dataset_name}/" + str(k).split("'")[1]
        if "Phase" in str(k):
            attributes[k]["unit"] = ("m", "m", "deg")
        elif "Amplitude" in str(k):
            attributes[k]["unit"] = ("m", "m", "V")
        elif "Height" in str(k):
            attributes[k]["unit"] = ("m", "m", "m")
        else:
            attributes[k]["unit"] = ("m", "m", "unknown")
    return data, metadata, attributes
