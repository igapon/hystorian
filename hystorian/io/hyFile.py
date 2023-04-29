import inspect
import types
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np

from . import gsf_files, ibw_files


class HyFile:
    """_summary_

    Attributes
    -------
    path : Path
        Path to the hdf5 file to be manipulated.
    file : h5py.File
        Handle of the file: the class call h5py.File(path). If the file does not exist, it is generated. (See __init__ docstring for more details).
    attrs : Attributes
        Attributes is an internal class, which allow the manipulation of hdf5 attributes through HyFile.

    Methods
        read()
        extract_data()
        apply()
    ------
    """

    class Attributes:
        def __init__(self, file: h5py.File):
            self.file = file

        def __contains__(self, path: str) -> bool:
            return path in self.file

        def __getitem__(self, path: Optional[str] = None) -> dict:
            if path is None:
                f = self.file
            else:
                f = self.file[path]

            return {key: f.attrs[key] for key in f.attrs.keys()}

        def __setitem__(self, path: str, attribute: tuple[str, Any]) -> None:
            key, val = attribute
            self.file[path].attrs[key] = val

    def __init__(self, path: Path | str, mode: str = "r"):
        """__init__ _summary_

        Parameters
        ----------
        path : Path | str
            Path to the file to be manipulated.
            If the file does not exist, a new hdf5 file is created with a root structure containing three hdf5 groups: 'datasets', 'metadata' and 'process'.
        mode : str, optional
            Mode in which the file should be opened, Valid modes are:
            * 'r': Readonly, file must exist. (default)
            * 'r+': Read/write, file must exist.
            * 'w': Create file, truncate if exists.
            * 'w-' or 'x': Create file, fail if exists
            * 'a' : Read/write if exists, create oterwise
        """
        self.path = Path(path)

        if self.path.is_file():
            self.file = h5py.File(self.path, mode)
            root_struct = set(self.file.keys())
            if root_struct != {"datasets", "metadata", "process"}:
                warnings.warn(
                    f"Root structure of the hdf5 file is not composed of 'datasets', 'metadata' and 'process'. \n It may not have been created with Hystorian. \n Current root structure is {root_struct}"
                )

        else:
            self.file = h5py.File(self.path, "a")
            for group in ["datasets", "metadata", "process"]:
                self._require_group(group)

            if mode != "a":
                self.file.close()
                self.file = h5py.File(self.path, mode)

        self.attrs = self.Attributes(self.file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback) -> bool:
        if self.file:
            self.file.close()

        if value is not None:
            warnings.warn(f"File closed with the following error: {exc_type} - {value} \n {traceback}")
            return False

        return True

    def __getitem__(self, path: str = "") -> list[str] | h5py.Group | h5py.Dataset | h5py.Datatype:
        if path == "":
            return list(self.file.keys())
        else:
            return self.file[path]

    def __setitem__(self, data: tuple[str, Any], overwrite=True) -> None:
        self._create_dataset(data, overwrite)

    def __delitem__(self, path: str) -> None:
        if path not in self.file:
            raise KeyError(f"Path {path} does not exist in the file.")
        del self.file[path]

    def __contains__(self, path: str) -> bool:
        return path in self.file

    def read(self, path: Optional[str] = None) -> list[str] | h5py.Dataset | h5py.Datatype:
        """read
        Wrapper around the __getitem__ of h5py. Directly returns the keys of the sub-groups if the path lead to an h5py.Group, otherwise directly load the dataset.
        This allow to get the keys to the folders without calling .keys(), therefore the way to call the keys or the data are the same.

        Parameters
        ----------
        path : Optional[str], optional
            _description_, by default None

        Returns
        -------
        list[str] | h5py.Dataset | h5py.Datatype
            _description_
        """
        if path is None:
            return list(self.file.keys())
        else:
            current = self.file[path]
            if isinstance(current, h5py.Group):
                return list(current.keys())
            if isinstance(current, h5py.Datatype):
                return current
            else:
                return current[()]

    def extract_data(self, path: str | Path):
        conversion_functions = {
            ".gsf": gsf_files.extract_gsf,
            ".ibw": ibw_files.extract_ibw,
        }

        if isinstance(path, str):
            path = Path(path)

        if path.suffix in conversion_functions:
            data, metadata, attributes = conversion_functions[path.suffix](path)
            self._write_extracted_data(path, data, metadata, attributes)
        else:
            raise TypeError(f"{path.suffix} file doesn't have a conversion function.")

    def apply(
        self,
        function: Callable,
        inputs: list[str] | str,
        output_names: Optional[list[str] | str] = None,
        increment_proc: bool = True,
        **kwargs: dict[str, Any],
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
            raise ValueError(
                f"Error: Unequal amount of outputs ({len(result)}) and output names ({len(output_names)})."
            )

        num_proc = len(self.read("process"))

        if increment_proc or self._generate_process_folder_name(num_proc, function) not in self.read("process"):
            num_proc += 1

        out_folder_location = self._generate_process_folder_name(num_proc, function)

        for name, data in zip(output_names, result):
            self._create_dataset((f"{out_folder_location}/{name}", data))

            self._write_generic_attributes(f"{out_folder_location}/{name}", inputs, name, function)
            self._write_kwargs_as_attributes(
                f"{out_folder_location}/{name}", function, kwargs, first_kwarg=len(inputs)
            )

    def _generate_process_folder_name(self, num_proc: int, function: Callable) -> str:
        return f"{str(num_proc).zfill(3)}-{function.__name__}"

    def _write_generic_attributes(
        self, out_folder_location: str, in_paths: list[str] | str, output_name: str, function: Callable
    ) -> None:
        if not isinstance(in_paths, list):
            in_paths = [in_paths]

        operation_name = out_folder_location.split("/")[1]
        new_attrs = {
            "path": out_folder_location + output_name,
            "shape": np.shape(self[out_folder_location]),
            "name": output_name,
        }

        new_attrs["operation name"] = (function.__module__ or "None") + "." + function.__name__

        if function.__module__ == "__main__":
            new_attrs["function code"] = inspect.getsource(function)

        new_attrs["operation number"] = operation_name.split("-")[0]
        new_attrs["time"] = str(datetime.now())
        new_attrs["source"] = in_paths

        self.attrs[out_folder_location] = new_attrs

    def _write_kwargs_as_attributes(
        self, path: str, func: Callable, all_variables: dict[str, Any], first_kwarg: int = 1
    ) -> None:
        attr_dict = {}
        if isinstance(func, types.BuiltinFunctionType):
            attr_dict["BuiltinFunctionType"] = True
        else:
            signature = inspect.signature(func).parameters
            var_names = list(signature.keys())[first_kwarg:]
            for key in var_names:
                if key in all_variables:
                    value = all_variables[key]
                elif isinstance(signature[key].default, np._globals._NoValueType):
                    value = "None"
                else:
                    value = signature[key].default

                if callable(value):
                    value = value.__name__
                elif value is None:
                    value = "None"

                try:
                    attr_dict[f"kwargs_{key}"] = value
                except RuntimeError:
                    RuntimeWarning("Attribute was not able to be saved, probably because the attribute is too large")
                    attr_dict[f"kwargs_{key}"] = "None"

        self.attrs[path] = attr_dict

    def _require_group(self, name: str):
        self.file.require_group(name)

    def _create_dataset(self, dataset: tuple[str, Any], overwrite=True) -> None:
        key, data = dataset
        if key in self.file:
            if overwrite:
                del self.file[key]
            else:
                raise KeyError("Key already exist and overwriste is set to False.")

        self.file.create_dataset(key, data=data)

    def _write_extracted_data(
        self, path: Path, data: dict[str, Any], metadata: dict[str, Any], attributes: dict[str, Any]
    ) -> None:
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
