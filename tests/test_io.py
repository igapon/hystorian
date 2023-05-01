# pylint: skip-file
import os
import pathlib
import unittest

import numpy as np
import pytest
from igor2 import binarywave

from hystorian.io import hyFile

filepath = pathlib.Path("tests/test_files/test.hdf5")


@pytest.fixture(autouse=True)
def teardown_module():
    if filepath.is_file():
        os.remove(filepath)


class TestHyFileClass:
    def test_init(self):
        with hyFile.HyFile(filepath) as f:
            root_keys = f[""]

        assert root_keys == ["datasets", "metadata", "process"]

    def test_double_init(self):
        with hyFile.HyFile(filepath) as f:
            pass

        with hyFile.HyFile(filepath) as f:
            root_keys = f[""]

        assert root_keys == ["datasets", "metadata", "process"]

    def test_write(self):
        with hyFile.HyFile(filepath, "r+") as f:
            f._create_dataset(("datasets/test_data", np.arange(100)))
            data = f.read("datasets/test_data")

        assert (data == np.arange(100)).all()

    def test_deep_write(self):
        with hyFile.HyFile(filepath, "r+") as f:
            f._create_dataset(("datasets/a/b/test_data", np.arange(100)))
            data = f.read("datasets/a/b/test_data")

        assert (data == np.arange(100)).all()

    def test_overwrite(self):
        with hyFile.HyFile(filepath, "r+") as f:
            f._create_dataset(("datasets/test_data", np.arange(100)))
            f._create_dataset(("datasets/test_data", np.arange(200)))

            data = f.read("datasets/test_data")

        assert (data == np.arange(200)).all()

        with pytest.raises(ValueError):
            f._create_dataset(("datasets/test_data", np.arange(200)), overwrite=False)

    def test_deletion(self):
        with hyFile.HyFile(filepath, "r+") as f:
            f._create_dataset(("datasets/test_data", np.arange(100)))
            del f["datasets/test_data"]

            with pytest.raises(KeyError):
                f["datasets/test_data"]

    def test_contain(self):
        with hyFile.HyFile(filepath) as f:
            assert "datasets" in f

            assert "not_a_folder" not in f


class TestHyFileConversion:
    def test_write_extracted_data(self):
        fake_path = "tmp_file"
        data = {"a": np.arange(100), "b": np.arange(200)}
        attr = {"a": {"name": "test_a", "size": 100}, "b": {"name": "test_b", "size": 200}}
        metadata = "This is metadata"
        with hyFile.HyFile(filepath, "r+") as f:
            f._write_extracted_data(pathlib.Path(fake_path), data, metadata, attr)

            assert f[f"datasets/{fake_path}/a"].attrs["name"] == "test_a"
            assert f[f"datasets/{fake_path}/b"].attrs["name"] == "test_b"
            assert f[f"datasets/{fake_path}/b"].attrs["size"] == 200
            assert (f[f"datasets/{fake_path}/b"] == np.arange(200)).all()

    """
    def test_extraction_ibw(self):
        path = pathlib.Path("test_files/raw_files/test_ibw.ibw")
        with hyFile.HyFile(filepath, "r+") as f:
            f.extract_data(path)

            tmpdata = binarywave.load(path)["wave"]
            metadata = tmpdata["note"]

            assert f.read(f"metadata/{path.stem}") == metadata
    """

    def test_extraction_gsf(self):
        path = pathlib.Path("tests/test_files/raw_files/test_gsf.gsf")
        with hyFile.HyFile(filepath, "r+") as f:
            f.extract_data(path)

            gsfFile = open(path, "rb")  # + ".gsf", "rb")

            metadata = {}

            # check if header is OK
            if not (gsfFile.readline().decode("UTF-8") == "Gwyddion Simple Field 1.0\n"):
                gsfFile.close()
                raise ValueError("File has wrong header")

            term = b"00"
            # read metadata header
            while term != b"\x00":
                line_string = gsfFile.readline().decode("UTF-8")
                metadata[line_string.rpartition("=")[0].strip()] = line_string.rpartition("=")[2].strip()
                term = gsfFile.read(1)

                gsfFile.seek(-1, 1)

            gsfFile.close()

            assert f.read(f"metadata/{path.stem}/XReal") == float(metadata["XReal"])


if __name__ == "__main__":
    unittest.main()
