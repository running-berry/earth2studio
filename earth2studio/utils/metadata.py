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


def create_dummy_metadata(
    variable: str | list[str] | VariableArray,
    conditioning_variable: str | list[str] | VariableArray = "",
    invariant: str | list[str] | VariableArray = "c",
    y: int = 32,
    x: int = 32,
) -> xr.Dataset:
    """Creates a dummy metadata xarray dataset with a structure matching the StormCast model's metadata.

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

    Returns
    -------
    xarray.Dataset
        A dummy Stormcast metadata xarray dataset

    """

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(conditioning_variable, str):
        # conditioning_variable = [conditioning_variable]
        conditioning_variable = []

    if isinstance(invariant, str):
        invariant = [invariant]

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

    data_vars = {
        "conditioning_means": (
            ("conditioning_variable",),
            np.random.rand(dims["conditioning_variable"]).astype(np.float32),
        ),
        "conditioning_stds": (
            ("conditioning_variable",),
            np.random.rand(dims["conditioning_variable"]).astype(np.float32) + 0.1,
        ),
        "invariants": (
            ("invariant", "y", "x"),
            np.random.rand(dims["invariant"], dims["y"], dims["x"]).astype(np.float32),
        ),
        "means": (("variable",), np.random.rand(dims["variable"]).astype(np.float32)),
        "stds": (
            ("variable",),
            np.random.rand(dims["variable"]).astype(np.float32) + 0.1,
        ),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    return ds
