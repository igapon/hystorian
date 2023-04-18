import h5py
import pathlib
import warnings
from . import ibw_files


# Why was this not commited ?
class hyFile:
    class Attributes:
        def __init__(self, file):
            self.file = file

        def __contains__(self, path: str) -> bool:
            return path in self.file

        def __getitem__(self, path: str | None = None) -> dict:
            if path is None:
                f = self.file[""]
            if path != "":
                f = self.file[path]

            return {key: f.attrs[key] for key in f.attrs.keys()}

        def __setitem__(self, path: str, attribute: tuple[str, any]) -> None:
            self.file[path].attrs[attribute[0]] = attribute[1]

    def __init__(self, path):
        if isinstance(path, str):
            self.path = pathlib.Path(path)
        else:
            self.path = path

        if not self.path.is_file():
            self.file = h5py.File(self.path, "a")
            for group in ["datasets", "metadata", "process"]:
                self._require_group(group)

        else:
            self.file = h5py.File(self.path, "r+")
            root_struct = set(self.file.keys())
            if root_struct != {"datasets", "metadata", "process"}:
                warnings.warn(
                    f"Root structure of the hdf5 file is not composed of 'datasets', 'metadata' and 'process'. \n It may not have been created with Hystorian."
                )

        self.attrs = self.Attributes(self.file)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.file:
            self.file.close

        if value is not None:
            warnings.warn(f"File closed with the following error: {type} - {value} \n {traceback}")

    def __getitem__(self, path: str = ""):
        if path == "":
            return self.file.keys()
        else:
            if isinstance(self.file[path], h5py.Group):
                return self.file[path].keys()
            else:
                return self.file[path]

    def __setitem__(self, data: tuple[str, any], overwrite=True):
        self._create_dataset(data, overwrite)

    def __delitem__(self, path: str):
        if path not in self.file:
            raise KeyError("Path does not exist in the file.")
        del self.file[path]

    def __contains__(self, path: str) -> bool:
        return path in self.file

    def _require_group(self, name: str):
        self.file.require_group(name)

    def _create_dataset(self, data: tuple[str, any], overwrite=True):
        if data[0] in self.file:
            if overwrite:
                del self.file[data[0]]
            else:
                raise KeyError("Key already exist and overwriste is set to False.")

        self.file.create_dataset(data[0], data=data[1])

    def extract_data(self, path):
        conversion_fcts = {"ibw": ibw_files.extract_ibw}

        if isinstance(path, str):
            path = pathlib.Path(path)

        if path.suffix in conversion_fcts:
            data, metadata, attributes = conversion_fcts[path.suffix()](path)
            self._write_extracted_data(path, data, metadata, attributes)

    def _write_extracted_data(self, path, data, metadata, attributes):
        self._require_group(f"datasets/{path.stem}")

        for d_key, d_value in data.items():
            self._create_dataset((f"datasets/{path.stem}/{d_key}", d_value), overwrite=True)

            for attr in attributes[d_key].items():
                self.attrs[f"datasets/{path.stem}/{d_key}"] = attr

        self._create_dataset((f"metadata/{path.stem}", metadata))
