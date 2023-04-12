import h5py


def create_empty_hdf5(filepath):
    with h5py.File(filepath.with_suffix(".hdf5"), "w") as f:
        f.create_group("metadata")
        f["metadata"].create_dataset(filepath.stem, dtype=h5py.Empty(h5py.string_dtype(encoding="utf-8")))

        f.create_group("process")

        datagrp = f.create_group(f"datasets/{filepath.stem}")
        datagrp.attrs["type"] = h5py.Empty(h5py.string_dtype(encoding="utf-8"))
