# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import xarray as xr

from earth2studio.utils.type import VariableArray


def create_metadata(
    variable: str | list[str] | VariableArray,
    conditioning_variable: str | list[str] | VariableArray,
    invariant: str | list[str] | VariableArray,
    y: int = 32,
    x: int = 32,
    variable_file_path: str | None = None,
    conditioning_variable_file_path: str | None = None,
    invariant_file_path: str | None = None,
) -> xr.Dataset:
    """Creates a StormCast metadata xarray dataset with a structure matching the StormCast model's metadata.

    Parameters
    ----------
    variable : str | list[str] | VariableArray
        String, list of strings or array of strings that refer to variables to
        return. Must be in the HRRR lexicon.
    conditioning_variable : str | list[str] | VariableArray
        String, list of strings or array of strings that refer to conditioning variables to
        return. Must be in the HRRR lexicon.
    invariant : str | list[str] | VariableArray
        String, list of strings or array of strings that refer to invariant to
        return. Must be in the HRRR lexicon.
    y : int
        The number of latitude grid points.
    x : int
        The number of longitude grid points.
    variable_file_path : str | None
        Path to the file containing the variable stds/means. If None, the variable stds/means will be randomly generated.
    conditioning_variable_file_path : str | None
        Path to the file containing the conditioning variable stds/means. If None, the conditioning variable stds/means will be randomly generated.
    invariant_file_path : str | None
        Path to the file containing the invariant data. If None, the invariant data will be randomly generated.

    Returns
    -------
    xarray.Dataset
        A Stormcast metadata xarray dataset

    """

    variable, conditioning_variable, invariant = prep_metadata_inputs(
        variable, conditioning_variable, invariant
    )

    dims = {
        "conditioning_variable": len(conditioning_variable),
        "invariant": len(invariant),
        "y": y,
        "x": x,
        "variable": len(variable),
    }

    coords = {
        "conditioning_variable": conditioning_variable,
        "invariant": invariant,
        "lat": (
            ("y", "x"),
            np.random.rand(dims["y"], dims["x"]).astype(np.float32) * 90,
        ),
        "lon": (
            ("y", "x"),
            np.random.rand(dims["y"], dims["x"]).astype(np.float32) * 180,
        ),
        "variable": variable,
        "x": np.arange(dims["x"]),
        "y": np.arange(dims["y"]),
    }

    variable_means, variable_stds = load_means_stds(
        (dims["variable"],), variable_file_path
    )

    conditioning_means, conditioning_stds = load_means_stds(
        (dims["conditioning_variable"],), conditioning_variable_file_path
    )

    invariant_data = load_invariant_data(
        (dims["invariant"], dims["y"], dims["x"]), invariant_file_path
    )

    data_vars = {
        "conditioning_means": (
            ("conditioning_variable",),
            conditioning_means,
        ),
        "conditioning_stds": (
            ("conditioning_variable",),
            conditioning_stds,
        ),
        "invariants": (("invariant", "y", "x"), invariant_data),
        "means": (("variable",), variable_means),
        "stds": (("variable",), variable_stds),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    return ds


def prep_metadata_inputs(
    variable: str | list[str] | VariableArray,
    conditioning_variable: str | list[str] | VariableArray,
    invariant: str | list[str] | VariableArray,
) -> tuple[list[str], list[str], list[str]]:
    """Simple method to pre-process metadata inputs into a common form

    Parameters
    ----------
    variable : str | list[str] | VariableArray
        String, list of strings or array of strings that refer to variables
    conditioning_variable : str | list[str] | VariableArray
        String, list of strings or array of strings that refer to conditioning variables
    invariant : str | list[str] | VariableArray
        String, list of strings or array of strings that refer to invariant variables

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Variable, conditioning variable, and invariant lists
    """
    if isinstance(variable, str):
        variable = [variable]

    if isinstance(conditioning_variable, str):
        conditioning_variable = [conditioning_variable]

    if isinstance(invariant, str):
        invariant = [invariant]

    return variable, conditioning_variable, invariant


def load_means_stds(
    dims: tuple[int, ...], file_path: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Simple method to load means and standard deviations from .npy files or generate random numpy arrays.

    Parameters
    ----------
    dims : tuple[int, ...]
        Dimensions of the array to be loaded or generated.
    file_path : str | None
        Path to the directory containing the means and stds .npy files. If None, random data will be generated.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the means and standard deviations as numpy arrays.
    """
    if file_path is not None:
        means = np.load(f"{file_path}/means.npy")
        stds = np.load(f"{file_path}/stds.npy")
    else:
        means = np.random.rand(*dims).astype(np.float32)
        stds = np.random.rand(*dims).astype(np.float32) + 0.1
    return means, stds


def load_invariant_data(dims: tuple[int, ...], file_path: str | None) -> np.ndarray:
    """Simple method to load invariant data from zarr files or generate random numpy arrays.

    Parameters
    ----------
    dims : tuple[int, ...]
        Dimensions of the array to be loaded or generated.
    file_path : str | None
        Path to the directory containing the invariant data files. If None, random data will be generated.

    Returns
    -------
    np.ndarray
        Numpy array containing the invariant data.
    """
    if file_path is not None:
        return xr.open_zarr(
            file_path
        ).values  # TODO: fix this after invariant data is available
    else:
        return np.random.rand(*dims).astype(np.float32)
