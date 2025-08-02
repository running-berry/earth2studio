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

import asyncio
import concurrent.futures
import functools
import os
import pathlib
import shutil
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import s3fs
import xarray as xr
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    prep_forecast_inputs,
)
from earth2studio.lexicon import HRRRFXLexicon, HRRRLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

try:
    import pyproj
except ImportError:
    OptionalDependencyFailure("data")
    pyproj = None

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

# Silence FutureWarning from cfgrib
warnings.simplefilter(action="ignore", category=FutureWarning)


@check_optional_dependencies()
class RWRF:
    """Radar Data Assimilation with WRFDA (RWRF) data source provides Taiwan weather analysis data developed by CWA. The radar observations possess high spatial resolution of approximately 1km and temporal resolutions of 5-10 minutes at a convective scale. The spatial dimensionality of RWRF data is [450, 450].

    Parameters
    ----------
    date_str : str, optional
        Date string of RWRF data in the format "YYYY/MM/DD", by default "2019/08/03"
    hr_str : int, optional
        Hour string of RWRF data in the range of 0-23, by default 0
    source_folder : str, optional
        Path to the folder containing RWRF data, by default "/home/master/14/andrewhsu/projects/physicsnemo/dev/data/rwrf"
    max_workers : int, optional
        Max works in async io thread pool. Only applied when using sync call function
        and will modify the default async loop if one exists, by default 24
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    TODO fix the whole RWRF description
    Additional information on the data repository can be referenced here:

    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    """

    RWRF_X = np.arange(450)
    RWRF_Y = np.arange(450)

    def __init__(
        self,
        date_str: str = "2019/08/03",
        hr_str: int = 0,
        source_folder: str = "/home/master/14/andrewhsu/projects/physicsnemo/dev/data/rwrf",
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        dt = datetime.strptime(date_str, "%Y/%m/%d")
        dt_str = dt.strftime(f"%Y-%m-%d_{str(hr_str).zfill(2)}")
        self._path = f"{source_folder}/{dt_str}/wrfout_d01_{dt_str}_interp"

        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers

        self.lexicon = HRRRLexicon
        self.async_timeout = async_timeout

        if os.path.exists(self._path):
            # TODO: fix the range after all RWRF data is available
            def _range(time: datetime) -> None:
                if time < datetime(year=2019, month=8, day=3):
                    raise ValueError(
                        f"Requested date time {time} needs to be after August 3rd, 2019 00:00 for RWRF"
                    )

            self._history_range = _range
        else:
            raise FileNotFoundError(f"Invalid RWRF source path {self._path}")

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve RWRF analysis data (lead time 0)

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to return. Must be in the RWRF lexicon.

        Returns
        -------
        xr.DataArray
            RWRF weather data array
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Modify the worker amount
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        xr_array = loop.run_until_complete(
            asyncio.wait_for(self.fetch(time, variable), timeout=self.async_timeout)
        )
        return xr_array

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the RWRF lexicon.

        Returns
        -------
        xr.DataArray
            RWRF weather data array
        """

        time, variable = prep_data_inputs(time, variable)
        logger.info(f"Stormcast time: {time}")
        logger.info(f"Stormcast vars: {variable}")
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Generate RWRF lat-lon grid to append onto data array
        lat, lon = self.grid()
        # Note, this could be more memory efficient and avoid pre-allocation of the array
        # but this is much much cleaner to deal with
        xr_array = xr.DataArray(
            data=np.zeros(
                (
                    len(time),
                    1,
                    len(variable),
                    len(self.RWRF_Y),
                    len(self.RWRF_X),
                )
            ),
            dims=["time", "lead_time", "variable", "rwrf_y", "rwrf_x"],
            coords={
                "time": time,
                "lead_time": [timedelta(hours=0)],
                "variable": variable,
                "rwrf_x": self.RWRF_X,
                "rwrf_y": self.RWRF_Y,
                "lat": (("rwrf_y", "rwrf_x"), lat),
                "lon": (("rwrf_y", "rwrf_x"), lon),
            },
        )

        ds = xr.open_dataset(self._path)

        var_map = {
            "t2m": "T2",
            "u10m": "umet10",
            # add any others here...
        }

        # # Loop through each requested variable and fill the xr_array
        for j, var in enumerate(variable):
            if var in var_map:
                data_slice = ds.variables[var_map.get(var, var)].isel(Time=0).values
                xr_array[0, 0, j, :, :] = data_slice
            else:
                logger.warning(
                    f"Variable '{var}' not found in {self._path}. Filling with random numbers."
                )
                xr_array[0, 0, j, :, :] = np.random.rand(
                    len(self.RWRF_Y), len(self.RWRF_X)
                )

        if not self._cache:
            shutil.rmtree(self.cache)

        xr_array = xr_array.isel(lead_time=0)
        del xr_array.coords["lead_time"]
        return xr_array

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify if date time is valid for RWRF based on offline knowledge

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            # TODO: fix this method after knowing RWRF intervals
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for HRRR"
                )
            # Check history range for given path
            self._history_range(time)

    def get_global_attrs(
        self,
    ) -> dict:
        """Get the global attributes of the RWRF dataset

        Returns
        -------
        dict
            Dictionary of global attributes from the dataset.
        """
        ds = xr.open_dataset(self._path, decode_coords=True, mask_and_scale=False)
        logger.debug("RWRF dataset global attributes:")
        for k, v in ds.attrs.items():
            logger.debug(f"{k}: {v}")

        return ds.attrs

    def get_lat_lon_bound(self) -> dict[str, tuple[float, float]]:
        """Get the NW, SW, SE, NE latitude and longitude corners of the RWRF dataset.

        Returns
        -------
        dict[str, tuple[float, float]]
            Dictionary with keys 'NW', 'SW', 'SE', 'NE' and values (lat, lon)
        """
        ds = xr.open_dataset(self._path, decode_coords=True, mask_and_scale=False)
        lat = ds.variables["XLAT"][0, ...]
        lon = ds.variables["XLONG"][0, ...]

        corners = {
            "SW": (lat[0, 0].item(), lon[0, 0].item()),
            "NW": (lat[-1, 0].item(), lon[-1, 0].item()),
            "NE": (lat[-1, -1].item(), lon[-1, -1].item()),
            "SE": (lat[0, -1].item(), lon[0, -1].item()),
        }
        logger.debug(f"RWRF corners: {corners}")
        return corners

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "rwrf")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_rwrf")
        return cache_location

    @classmethod
    def grid(cls) -> tuple[np.array, np.array]:
        """Generates the RWRF lambert conformal projection grid coordinates. Creates the RWRF grid using a secant (parallel) Lambert conformal conic projection with two standard parallels for accurate mapping over Taiwan.

        Note
        ----
        For more information about the RWRF grid see:

        - examples/09-1_stormcast_example.py:135

        Returns
        -------
        Returns:
            tuple: (lat, lon) in degrees
        """
        # lon_0, lat_0 is the center of the grid
        # lat_1, lat_2 is the standard parallel
        # a, b is radius of globe 6371229
        # RWRF().get_global_attrs()  # Print global attributes for debugging
        p1 = pyproj.CRS(
            "proj=lcc lon_0=120.0 lat_0=21.494176864624023 lat_1=10.0 lat_2=40.0 a=6371229 b=6371229"
        )
        p2 = pyproj.CRS("latlon")
        transformer = pyproj.Transformer.from_proj(p2, p1)
        itransformer = pyproj.Transformer.from_proj(p1, p2)

        # Start with getting grid bounds based on lat / lon box (SW-NW-NE-SE)
        # Ground-truth corner coordinates from the RWRF dataset [SW, NW, NE, SE]
        # Extracted from the provided XLAT and XLONG variables.
        # RWRF().get_lat_lon_bound()  # Print lat/lon bounds for debugging
        lat = np.array(
            [
                19.548282623291016,
                27.897096633911133,
                27.844585418701172,
                19.499420166015625,
            ]
        )
        lon = np.array(
            [
                116.37149047851562,
                116.11553955078125,
                125.56777954101562,
                125.20111083984375,
            ]
        )

        # Transform lat/lon corners to projected easting/northing coordinates
        easting, northing = transformer.transform(lat, lon)
        nx = len(cls.RWRF_X)
        ny = len(cls.RWRF_Y)
        E, N = np.meshgrid(
            np.linspace(easting[0], easting[2], nx),
            np.linspace(northing[0], northing[1], ny),
        )

        lat, lon = itransformer.transform(E, N)  # Transform the projected grid back
        lon = np.where(lon < 0, lon + 360, lon)
        return lat, lon

    @classmethod
    def get_corresponding_indices(
        cls, latitude: float, longitude: float, resolution: tuple[int, int] = (32, 32)
    ) -> tuple[int, int, int, int]:
        """Get the corresponding indices for the given latitude and longitude.

        Parameters
        ----------
        latitude : float
            Latitude of the point.
        longitude : float
            Longitude of the point.

        Returns
        -------
        tuple[int, int, int, int]
            A tuple containing the minimum and maximum indices for corresponding latitude and longitude in the RWRF grid.
        """

        rwrf_lat, rwrf_lon = cls.grid()

        dist = np.sqrt((rwrf_lat - latitude) ** 2 + (rwrf_lon - longitude) ** 2)
        y_index, x_index = np.unravel_index(np.argmin(dist), dist.shape)
        return (
            y_index - resolution[0] // 2,
            y_index + resolution[0] // 2,
            x_index - resolution[1] // 2,
            x_index + resolution[1] // 2,
        )

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in the HRRR object store. Uses S3
        store

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to access

        Returns
        -------
        bool
            If date time is avaiable
        """
        if isinstance(time, np.datetime64):  # np.datetime64 -> datetime
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        fs = s3fs.S3FileSystem(anon=True)
        # Object store directory for given time
        # Just picking the first variable to look for
        file_name = f"hrrr.{time.year}{time.month:0>2}{time.day:0>2}/conus"
        file_name = f"{file_name}/hrrr.t{time.hour:0>2}z.wrfnatf00.grib2.idx"
        s3_uri = f"s3://{cls.HRRR_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)

        return exists


class RWRF_FX(RWRF):
    """High-Resolution Rapid Refresh (HRRR) forecast source provides a North-American
    weather forecasts with hourly forecast runs developed by NOAA. This forecast source
    has hourly forecast steps up to a lead time of 48 hours. Data is provided on a
    Lambert conformal 3km grid at 1-hour intervals. The spatial dimensionality of HRRR
    data is [1059, 1799].

    Parameters
    ----------
    source : str, optional
        Data source to use ('aws', 'google', 'azure', 'nomads'), by default 'aws'
    max_workers : int, optional
        Max works in async io thread pool. Only applied when using sync call function
        and will modify the default async loop if one exists, by default 24
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    48 hour forecasts are provided on 6 hour intervals. 18 hour forecasts are generated
    hourly.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://rapidrefresh.noaa.gov/hrrr/
    - https://console.cloud.google.com/marketplace/product/noaa-public/hrrr
    """

    def __init__(
        self,
        source: str = "aws",
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        super().__init__(
            source=source,
            max_workers=max_workers,
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )
        self.lexicon = HRRRFXLexicon  # type: ignore

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve HRRR forecast data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        lead_time: timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HRRR lexicon.

        Returns
        -------
        xr.DataArray
            HRRR forecast data array
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Modify the worker amount
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        xr_array = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, lead_time, variable), timeout=self.async_timeout
            )
        )

        return xr_array

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        lead_time: timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HRRR FX lexicon.

        Returns
        -------
        xr.DataArray
            HRRR forecast data array
        """
        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)
        self._validate_leadtime(time, lead_time)

        # https://filesystem-spec.readthedocs.io/en/latest/async.html#using-from-async
        if isinstance(self.fs, s3fs.S3FileSystem):
            session = await self.fs.set_session()
        else:
            session = None

        # Note, this could be more memory efficient and avoid pre-allocation of the array
        # but this is much much cleaner to deal with, compared to something seen in the
        # NCAR data source.
        xr_array = xr.DataArray(
            data=np.empty(
                (
                    len(time),
                    len(lead_time),
                    len(variable),
                    len(self.RWRF_Y),
                    len(self.RWRF_X),
                )
            ),
            dims=["time", "lead_time", "variable", "hrrr_y", "hrrr_x"],
            coords={
                "time": time,
                "lead_time": lead_time,
                "variable": variable,
                "hrrr_x": self.RWRF_X,
                "hrrr_y": self.RWRF_Y,
            },
        )

        async_tasks = []
        async_tasks = await self._create_tasks(time, lead_time, variable)
        func_map = map(
            functools.partial(self.fetch_wrapper, xr_array=xr_array), async_tasks
        )

        await tqdm.gather(
            *func_map, desc="Fetching HRRR data", disable=(not self._verbose)
        )

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        # Close aiohttp client if s3fs
        if session:
            await session.close()

        return xr_array

    @classmethod
    def _validate_leadtime(
        cls, times: list[datetime], lead_times: list[timedelta]
    ) -> None:
        """Verify if lead time is valid for HRRR based on offline knowledge

        Parameters
        ----------
        lead_times : list[timedelta]
            list of lead times to fetch data
        """
        for time in times:
            for delta in lead_times:
                if not delta.total_seconds() % 3600 == 0:
                    raise ValueError(
                        f"Requested lead time {delta} needs to be 1 hour interval for HRRR"
                    )
                hours = int(delta.total_seconds() // 3600)
                # Note, one forecasts every 6 hours have 2 day lead times, others only have 18 hours
                if hours > 48 or hours < 0:
                    raise ValueError(
                        f"Requested lead time {delta} can only be between [0,48] hours for HRRR forecast"
                    )
                if (
                    not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0
                    and hours > 18
                ):
                    raise ValueError(
                        f"Requested lead time {delta} can only be between [0,18] hours for HRRR forecast not on 6 hour interval {time}"
                    )
