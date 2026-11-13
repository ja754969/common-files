"""Spatial masking helpers for two-dimensional latitude/longitude grids."""

from __future__ import annotations

import xarray as xr


def select_curvilinear_region(
    da: xr.DataArray,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    convention: str,
) -> xr.DataArray:
    """Mask a two-dimensional latitude/longitude grid without 1-D indexing."""
    longitude = da.lon
    if float(longitude.max()) > 180:
        longitude = ((longitude + 180) % 360) - 180
        da = da.assign_coords(lon=longitude)
    lon_min, lon_max = lon_bounds
    internal_min = lon_min if convention == "-180 to 180" else ((lon_min + 180) % 360) - 180
    internal_max = lon_max if convention == "-180 to 180" else ((lon_max + 180) % 360) - 180
    is_global = convention == "0 to 360" and lon_min == 0 and lon_max == 360
    if is_global:
        longitude_mask = xr.ones_like(longitude, dtype=bool)
    elif internal_min <= internal_max:
        longitude_mask = (longitude >= internal_min) & (longitude <= internal_max)
    else:
        longitude_mask = (longitude >= internal_min) | (longitude <= internal_max)
    latitude_mask = (da.lat >= min(lat_bounds)) & (da.lat <= max(lat_bounds))
    result = da.where(longitude_mask & latitude_mask, drop=True)
    if not result.coords["lon"].size or not result.coords["lat"].size:
        raise ValueError("The selected region does not overlap the dataset.")
    return result
