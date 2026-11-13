"""Memory-safe grid-cell linear detrending for xarray data."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _decimal_year(time_values: np.ndarray) -> np.ndarray:
    """Convert datetime64 values to a fractional Gregorian year."""
    dates = np.asarray(time_values).astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    year_start = dates.astype("datetime64[Y]").astype("datetime64[D]")
    day_of_year = (dates - year_start).astype("timedelta64[D]").astype(float)
    leap_year = ((years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))).astype(float)
    return years.astype(float) + day_of_year / (365.0 + leap_year)


def detrend_data(
    da: xr.DataArray,
    detrend_type: str = "normalize",
    reference_year: float | None = None,
) -> xr.DataArray:
    """Remove a grid-cell linear trend without materialising the full field.

    ``normalize`` preserves the fitted value at ``reference_year``; ``anomaly``
    returns residuals from the fitted line. Missing samples remain missing;
    a cell is fitted when it has at least two finite samples.
    """
    if detrend_type not in {"normalize", "anomaly"}:
        raise ValueError("detrend_type must be 'normalize' or 'anomaly'.")
    if "time" not in da.dims:
        raise ValueError("Detrending requires a time dimension.")

    if da.sizes["time"] < 2:
        raise ValueError("Detrending requires at least two time samples.")
    series = da.transpose(*[dim for dim in da.dims if dim != "time"], "time")
    time_values = np.asarray(series.time.values)
    if np.issubdtype(time_values.dtype, np.datetime64):
        time_axis = _decimal_year(time_values)
    else:
        time_axis = time_values.astype(float)
    if reference_year is None:
        reference_year = float(np.median(time_axis))

    # Closed-form least-squares sums keep the operation lazy while fitting each
    # cell from its available finite samples.
    time_coordinate = xr.DataArray(time_axis, dims="time")
    finite = xr.apply_ufunc(np.isfinite, series, dask="allowed", output_dtypes=[bool])
    safe_values = series.where(finite, 0.0)
    valid_count = finite.sum("time")
    safe_time = xr.where(finite, time_coordinate, 0.0)
    mean_time = safe_time.sum("time") / valid_count
    mean_value = safe_values.sum("time") / valid_count
    centered_time = time_coordinate - mean_time
    denominator = (finite * centered_time**2).sum("time")
    valid_fit = (valid_count >= 2) & (denominator > 0)
    if np.isclose(float(np.asarray(denominator.max())), 0.0):
        raise ValueError("Detrending requires at least two distinct time values.")
    slope = (
        (finite * (safe_values - mean_value) * centered_time).sum("time") / denominator
    ).where(valid_fit)
    trend = slope * (time_coordinate - reference_year)
    if detrend_type == "anomaly":
        # The intercept form is algebraically equivalent to y - fitted trend.
        fitted = slope * (time_coordinate - mean_time) + mean_value
        result = (series - fitted).where(finite & valid_fit)
    else:
        result = (series - trend).where(finite & valid_fit)
    result = result.transpose(*da.dims)
    result.name = da.name
    result.attrs = da.attrs.copy()
    result.attrs["detrend_type"] = detrend_type
    result.attrs["detrend_reference_year"] = reference_year
    result.attrs["detrend_note"] = "Grid-cell linear detrending applied before temporal aggregation."
    return result
