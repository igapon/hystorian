import unittest
import h5py
import os
import pathlib
import io
import numpy as np
import pytest

from hystorian.io import hyFile


filepath = pathlib.Path("test_files/test.hdf5")


@pytest.fixture(autouse=True)
def teardown_module():
    if filepath.is_file():
        os.remove(filepath)


class TestHyFileClass:
    def test_init(self):
        with hyFile.hyFile(filepath) as f:
            root_keys = list(f[""])

        assert root_keys == ["datasets", "metadata", "process"]

    def test_double_init(self):
        with hyFile.hyFile(filepath) as f:
            pass

        with hyFile.hyFile(filepath) as f:
            root_keys = list(f[""])

        assert root_keys == ["datasets", "metadata", "process"]

    def test_write(self):
        with hyFile.hyFile(filepath) as f:
            f._create_dataset(("datasets/test_data", np.arange(100)))
            data = f["datasets/test_data"][()]

        assert (data == np.arange(100)).all()

    def test_deep_write(self):
        with hyFile.hyFile(filepath) as f:
            f._create_dataset(("datasets/a/b/test_data", np.arange(100)))
            data = f["datasets/a/b/test_data"]

        assert (data == np.arange(100)).all()

    def test_overwrite(self):
        with hyFile.hyFile(filepath) as f:
            f._create_dataset(("datasets/test_data", np.arange(100)))
            f._create_dataset(("datasets/test_data", np.arange(200)))

            data = f["datasets/test_data"]

        assert (data == np.arange(200)).all()

        with pytest.raises(KeyError):
            f._create_dataset(("datasets/test_data", np.arange(200)), overwrite=False)

    def test_deletion(self):
        with hyFile.hyFile(filepath) as f:
            f._create_dataset(("datasets/test_data", np.arange(100)))
            del f["datasets/test_data"]

            with pytest.raises(KeyError):
                f["datasets/test_data"]

    def test_contain(self):
        with hyFile.hyFile(filepath) as f:
            assert "datasets" in f

            assert "not_a_folder" not in f

    def test_write_extracted_data(self):
        fake_path = "tmp_file"
        data = {"a": np.arange(100), "b": np.arange(200)}
        attr = {"a": {"name": "test_a", "size": 100}, "b": {"name": "test_b", "size": 200}}
        metadata = "This is metadata"
        with hyFile.hyFile(filepath) as f:
            f._write_extracted_data(pathlib.Path(fake_path), data, metadata, attr)

            assert f[f"datasets/{fake_path}/a"].attrs["name"] == "test_a"
            assert f[f"datasets/{fake_path}/b"].attrs["name"] == "test_b"
            assert f[f"datasets/{fake_path}/b"].attrs["size"] == 200
            assert (f[f"datasets/{fake_path}/b"] == np.arange(200)).all()

    def test_extraction(self):
        path = pathlib.Path("/test_files/raw_files/test_ibw.ibw")
        with hyFile.hyFile(filepath) as f:
            f.extract_data(path)


if __name__ == "__main__":
    unittest.main()
