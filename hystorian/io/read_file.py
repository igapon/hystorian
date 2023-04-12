import os
import re

import h5py
import numpy as np

from . import ardf_files
from . import csv_files
from . import gsf_files
from . import ibw_files
from . import img_files
from . import nanoscope_files
from . import shg_files
from . import sxm_files
from . import xrdml_files


def to_hdf5(filename, filepath=None, params=None):
    file_extension = filename.split(".")[-1]

    funct_dict = {
        "ardf": ardf_files.ardf2hdf5,
        "csv": csv_files.csv2hdf5,
        "gsf": gsf_files.gsf2hdf5,
        "ibw": ibw_files.ibw2hdf5,
        "sxm": sxm_files.sxm2hdf5,
        "xrdml": xrdml_files.xrdml2hdf5,
        "jpg": img_files.image2hdf5,
        "png": img_files.image2hdf5,
        "jpeg": img_files.image2hdf5,
        "bmp": img_files.image2hdf5,
    }

    if file_extension in funct_dict:
        if file_extension == "dat":
            try:
                funct_dict[file_extension](filename, filepath)
            except:
                raise Exception(
                    "This dat file has a shape which is not supported, for the moment dat file are used for SHG datas"
                )

        else:
            funct_dict[file_extension](filename, filepath)

    elif re.match("\d{3}", filename.split(".")[-1]) is not None:
        try:
            nanoscope_files.nanoscope2hdf5(filename, filepath)
        except:
            raise Exception(
                "Your file is ending with three digits, but is not a Nanoscope file, please change the "
                "extension to the correct one."
            )

    return file_extension
