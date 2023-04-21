import h5py
import pathlib
import warnings
from . import ibw_files, gsf_files


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

    def __exit__(self, type, value, traceback) -> None:
        if self.file:
            self.file.close

        if value is not None:
            warnings.warn(f"File closed with the following error: {type} - {value} \n {traceback}")

    def __getitem__(self, path: str = "") -> h5py.Group | h5py.Dataset:
        if path == "":
            return self.file
        else:
            return self.file[path]

    def __setitem__(self, data: tuple[str, any], overwrite=True) -> None:
        self._create_dataset(data, overwrite)

    def __delitem__(self, path: str) -> None:
        if path not in self.file:
            raise KeyError("Path does not exist in the file.")
        del self.file[path]

    def __contains__(self, path: str) -> bool:
        return path in self.file

    def read(self, path: str = ""):
        if path == "":
            return self.file.keys()
        else:
            if isinstance(self.file[path], h5py.Group):
                return self.file[path].keys()
            else:
                return self.file[path][()]

    def extract_data(self, path: str):
        conversion_fcts = {
            ".gsf": gsf_files.extract_gsf,
            ".ibw": ibw_files.extract_ibw,
        }

        if isinstance(path, str):
            path = pathlib.Path(path)

        if path.suffix in conversion_fcts:
            data, metadata, attributes = conversion_fcts[path.suffix](path)
            self._write_extracted_data(path, data, metadata, attributes)
        else:
            raise TypeError(f"{path.suffix} file don't have a conversion function.")

    def apply(
        self,
        function: callable,
        inputs: list[str] | str,
        output_names: list[str] | str | None = None,
        increment_proc: bool = True,
        **kwargs,
    ):
        def convert_to_list(inputs):
            if isinstance(inputs, list):
                return inputs
            return [inputs]

        inputs = convert_to_list(inputs)

        if output_names is None:
            output_names = inputs[0].rsplit("/", 1)[1]
        output_names = convert_to_list(output_names)

        result = function(*inputs, **kwargs)

        if result is None:
            return None
        if not isinstance(result, tuple):
            result = tuple([result])

        if len(output_names) != len(result):
            raise Exception("Error: Unequal amount of outputs and output names")

        num_proc = len(self.read("process"))

        if increment_proc or f"{str(num_proc).zfill(3)}-{function.__name__}" not in self.read("process"):
            num_proc += 1

        out_folder_location = f"{'process'}/{str(num_proc).zfill(3)}-{function.__name__}"

        for name, data in zip(output_names, result):
            self._create_dataset((f"{out_folder_location}/{name}", data))

    def _require_group(self, name: str):
        self.file.require_group(name)

    def _create_dataset(self, data: tuple[str, any], overwrite=True) -> None:
        if data[0] in self.file:
            if overwrite:
                del self.file[data[0]]
            else:
                raise KeyError("Key already exist and overwriste is set to False.")

        self.file.create_dataset(data[0], data=data[1])

    def _write_extracted_data(self, path, data, metadata, attributes) -> None:
        self._require_group(f"datasets/{path.stem}")

        for d_key, d_value in data.items():
            self._create_dataset((f"datasets/{path.stem}/{d_key}", d_value), overwrite=True)

            for attr in attributes[d_key].items():
                self.attrs[f"datasets/{path.stem}/{d_key}"] = attr

        if isinstance(metadata, dict):
            for key in metadata:
                self._create_dataset((f"metadata/{path.stem}/{key}", metadata[key]))
        else:
            self._create_dataset((f"metadata/{path.stem}", metadata))
