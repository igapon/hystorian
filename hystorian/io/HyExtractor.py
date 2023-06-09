import importlib
import re
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable

from .utils import check_extension

extractor_registery = {}


def initialize():
    for import_path, regex in zip(
        [".ibw_files", ".ardf_files", ".nanoscope_files", ".gsf_files"],
        [r"\.ibw", r"(?i).ARDF", r".\d{3}", r"\.gsf"],
    ):
        success, imported_package = _dynamic_import(import_path, package="hystorian.io.extractors")
        if success:
            extractor_registery[import_path] = (partial(check_extension, regex), imported_package.extract)  # type: ignore


def _dynamic_import(import_path, package=None):
    try:
        if package:
            imported_package = importlib.import_module(import_path, package)
        else:
            imported_package = importlib.import_module(import_path)
    except Exception as err:
        warnings.warn(
            f"{type(err)} {err} \n {import_path} was not able to be imported. You will not be able to convert files using this import."
        )
        return False, None

    return True, imported_package


def add_converter(name: str, f_check: Callable, f_converter: Callable):
    if name not in extractor_registery:
        extractor_registery[name] = (f_check, f_converter)
    else:
        warnings.warn(
            "A converter already exist under this name. Use another name or remove it using 'remove_converter'."
        )


def remove_converter(name: str):
    if name in extractor_registery:
        extractor_registery.pop(name)


def extract(filepath):
    extractor_lst = []
    for value in extractor_registery.values():
        check, f_convert = value
        if check(filepath):
            extractor_lst.append(f_convert)

    if len(extractor_lst) == 0:
        raise ValueError(f"No conversion function was found for {filepath}.")
    if len(extractor_lst) > 1:
        warnings.warn(
            "Multiple conversion functions were found for this file. {converter_lst[0].__name__} will be used.",
            category=UserWarning,
        )

    return extractor_lst[0](filepath)


initialize()
