import os
import pathlib
import unittest

import numpy as np
import pytest
from igor2 import binarywave

from hystorian.io import hyFile, utils

filepath = pathlib.Path("tests/test_files/test.hdf5")


class TestApply:
    def test_apply(self):
        with hyFile.HyFile(filepath, "r+") as f:
            assert hyFile.HyApply(f, np.sum, ([0, 1, 2, 3],)).apply() == (6, ([0, 1, 2, 3],), {})

    def test_apply_HyPath(self):
        path = pathlib.Path("tests/test_files/raw_files/test_nanonis.000")
        with hyFile.HyFile(filepath, "r+") as f:
            f.extract_data(path)

            f.apply(np.nansum, hyFile.HyPath("datasets/test_nanonis/HeightRetrace"), output_names=["test1"])
            f.apply(np.nansum, hyFile.HyPath("datasets/test_nanonis/HeightRetrace"), output_names=["test1"], axis=0)
            f.apply(
                np.nansum,
                [
                    hyFile.HyPath("datasets/test_nanonis/HeightRetrace"),
                    hyFile.HyPath("datasets/test_nanonis/AmplitudeRetrace"),
                ],
                output_names=["test1"],
            )

    def test_multiple_apply(self):
        path = pathlib.Path("tests/test_files/raw_files/test_nanonis.000")
        with hyFile.HyFile(filepath, "r+") as f:
            f.extract_data(path)
            f.multiple_apply(
                np.nansum,
                [
                    hyFile.HyPath("datasets/test_nanonis/HeightRetrace"),
                    hyFile.HyPath("datasets/test_nanonis/AmplitudeRetrace"),
                ],
                output_names=["test1", "test2"],
            )
