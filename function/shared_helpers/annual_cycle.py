"""Reusable annual-cycle analysis for gridded climate data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path


@dataclass(frozen=True)
class TimeResolution:
    """Observed temporal sampling and a conservative resolution label."""

    name: str
    median_interval_days: float
    regular: bool
    samples: int


@dataclass(frozen=True)
class AnnualCycleResult:
    """Annual-cycle curves and diagnostics ready for plotting."""

    resolution: str
    x: np.ndarray
    target: dict[int, np.ndarray]
    other: dict[int, np.ndarray]
    climatology: np.ndarray
    labels: tuple[str, ...]
    valid_fraction: dict[int, np.ndarray]
    warnings: tuple[str, ...]
    month_tick_positions: np.ndarray
    month_tick_labels: tuple[str, ...]
    day_tick_positions: np.ndarray
    day_tick_labels: tuple[str, ...]
    anchor_position: float


def detect_time_resolution(time: xr.DataArray | Sequence[object]) -> TimeResolution:
    """Classify a time coordinate by its observed sampling interval."""
    values = pd.DatetimeIndex(pd.to_datetime(np.asarray(time), errors="coerce")).dropna().unique().sort_values()
    if len(values) < 2:
        return TimeResolution("unknown", float("nan"), True, len(values))
    intervals = np.diff(values.asi8) / 86_400_000_000_000.0
    median = float(np.median(intervals))
    regular = bool(np.all(np.abs(intervals - median) <= max(0.05, median * 0.1)))
    if median < 1.0:
        name = "sub-daily"
    elif median <= 2.0:
        name = "daily"
    elif 2.0 < median < 27.0:
        rounded_days = int(round(median))
        name = f"{rounded_days}-day"
    elif 27.0 <= median <= 32.0:
        name = "monthly"
    elif 300.0 <= median <= 400.0:
        name = "yearly"
    else:
        name = "irregular"
    return TimeResolution(name, median, regular, len(values))


def polygon_grid_mask(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    vertices: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Return a boundary-inclusive mask for a polygon on a rectilinear grid."""
    if len(vertices) < 3:
        raise ValueError("A polygon requires at least three vertices.")
    points = np.asarray(vertices, dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("Polygon vertices must be finite.")
    reference = points[0, 0]
    unwrapped_vertices = points.copy()
    unwrapped_vertices[:, 0] = reference + ((unwrapped_vertices[:, 0] - reference + 180.0) % 360.0) - 180.0
    longitude_grid, latitude_grid = np.meshgrid(
        reference + ((np.asarray(longitudes, dtype=float) - reference + 180.0) % 360.0) - 180.0,
        np.asarray(latitudes, dtype=float),
    )
    path = Path(unwrapped_vertices, closed=True)
    xy = np.column_stack((longitude_grid.ravel(), latitude_grid.ravel()))
    mask = path.contains_points(xy, radius=1e-10).reshape(longitude_grid.shape)
    return mask


def spatial_mean_timeseries(
    data: xr.DataArray,
    vertices: Sequence[tuple[float, float]],
    weighting: str = "coslat",
) -> xr.DataArray:
    """Average a field inside a polygon for every time sample."""
    if not {"lat", "lon"}.issubset(data.dims):
        raise ValueError("Annual-cycle averaging requires lat and lon dimensions.")
    vertices_array = np.asarray(vertices, dtype=float)
    reference = vertices_array[0, 0]
    unwrapped_longitudes = reference + (
        (vertices_array[:, 0] - reference + 180.0) % 360.0
    ) - 180.0
    longitude_values = np.asarray(data.lon.values, dtype=float)
    longitude_unwrapped = reference + (
        (longitude_values - reference + 180.0) % 360.0
    ) - 180.0
    longitude_keep = (
        (longitude_unwrapped >= np.min(unwrapped_longitudes))
        & (longitude_unwrapped <= np.max(unwrapped_longitudes))
    )
    latitude_values = np.asarray(data.lat.values, dtype=float)
    latitude_keep = (
        (latitude_values >= np.min(vertices_array[:, 1]))
        & (latitude_values <= np.max(vertices_array[:, 1]))
    )
    if not longitude_keep.any() or not latitude_keep.any():
        raise ValueError("The polygon does not overlap the dataset.")
    data = data.isel(
        lon=np.flatnonzero(longitude_keep),
        lat=np.flatnonzero(latitude_keep),
    )
    mask = polygon_grid_mask(
        np.asarray(data.lat.values),
        np.asarray(data.lon.values),
        vertices,
    )
    selected = data.where(xr.DataArray(mask, dims=("lat", "lon")))
    if weighting == "coslat":
        weights = xr.DataArray(
            np.cos(np.deg2rad(selected.lat.values)),
            dims=("lat",),
            coords={"lat": selected.lat},
        )
        result = selected.weighted(weights).mean(("lat", "lon"), skipna=True)
    elif weighting == "equal":
        result = selected.mean(("lat", "lon"), skipna=True)
    else:
        raise ValueError("weighting must be 'coslat' or 'equal'.")
    return result


def continuous_running_mean(series: xr.DataArray, window: int, resolution: str) -> xr.DataArray:
    """Apply a running mean separately to contiguous time segments.

    Segmenting at large temporal gaps prevents a rolling window from silently
    bridging missing observations.  The operation is performed before annual
    cycle grouping, so December and January remain continuous.
    """
    if window < 1:
        raise ValueError("The running-mean window must be at least one.")
    if window == 1 or "time" not in series.dims:
        return series
    values = np.asarray(series.values, dtype=float)
    times = pd.DatetimeIndex(pd.to_datetime(series.time.values))
    if len(times) < 2:
        return series
    expected = {"sub-daily": 1.0, "daily": 1.0, "monthly": 31.0, "yearly": 366.0}.get(resolution)
    if expected is None:
        expected = float(np.median(np.diff(times.asi8) / 86_400_000_000_000.0))
    gaps = np.diff(times.asi8) / 86_400_000_000_000.0
    breaks = np.flatnonzero(gaps > max(expected * 1.75, expected + 1.0)) + 1
    segments = np.split(np.arange(len(times)), breaks)
    output = np.full(values.shape, np.nan, dtype=float)
    for segment in segments:
        frame = pd.Series(values[segment], index=times[segment])
        output[segment] = frame.rolling(window=window, min_periods=1, center=True).mean().to_numpy()
    return series.copy(data=output)


def build_annual_cycles(
    series: xr.DataArray,
    target_years: Iterable[int],
    baseline_years: tuple[int, int] | None,
    other_years: tuple[int, int] | None,
    anchor_month: int = 1,
    running_mean_window: int = 1,
    month_tick_position: str = "start",
) -> AnnualCycleResult:
    """Build continuous, sample-aware target, comparison, and climatology curves.

    ``month_tick_position`` controls which calendar day is used when locating
    month labels on fixed multi-day and sub-daily samples. The nearest
    observed sample is used, so the option remains valid for 8-, 11-, 15-day,
    and other fixed-interval products.
    """
    if "time" not in series.coords:
        raise ValueError("Annual-cycle data requires a time coordinate.")
    resolution_info = detect_time_resolution(series.time)
    if resolution_info.name == "yearly":
        raise ValueError("Yearly data cannot produce a monthly annual cycle.")
    smoothed = continuous_running_mean(series, running_mean_window, resolution_info.name)
    frame = pd.DataFrame({"value": np.asarray(smoothed.values, dtype=float)}, index=pd.DatetimeIndex(smoothed.time.values))
    frame["year"] = frame.index.year
    if resolution_info.name == "monthly":
        frame["slot"] = frame.index.month
        slot_values = np.arange(1, 13, dtype=int)
    else:
        if resolution_info.name.endswith("-day"):
            interval_days = int(resolution_info.name.split("-", 1)[0])
            frame["slot"] = ((frame.index.dayofyear - 1) // interval_days) * interval_days + 1
        else:
            frame["slot"] = frame.index.dayofyear + (
                (~frame.index.is_leap_year) & (frame.index.month > 2)
            )
        slot_values = np.sort(frame["slot"].dropna().unique().astype(int))
    if slot_values.size == 0:
        raise ValueError("Annual-cycle data contains no usable time samples.")
    if not 1 <= anchor_month <= 12:
        raise ValueError("anchor_month must be between 1 and 12.")
    if month_tick_position not in {"start", "middle"}:
        raise ValueError("month_tick_position must be 'start' or 'middle'.")
    month_tick_day = 1 if month_tick_position == "start" else 15
    anchor_slot = anchor_month if resolution_info.name == "monthly" else int(
        pd.Timestamp(2000, anchor_month, 1).dayofyear
    )
    start_index = int(np.argmin(np.abs(slot_values - anchor_slot)))
    one_cycle_order = np.concatenate(
        (np.arange(start_index, len(slot_values)), np.arange(0, start_index))
    )
    cycle_length = len(slot_values)
    x = np.arange(2 * cycle_length, dtype=float)
    month_tick_positions = []
    month_tick_labels = []
    for cycle_offset, suffix in ((0, "previous year"), (cycle_length, "target year")):
        for month in range(1, 13):
            month_slot = month if resolution_info.name == "monthly" else pd.Timestamp(
                2000, month, month_tick_day
            ).dayofyear
            nearest = int(np.argmin(np.abs(slot_values - month_slot)))
            month_tick_positions.append(float(cycle_offset + np.where(one_cycle_order == nearest)[0][0]))
            year_code = "-1" if suffix == "previous year" else "0"
            month_tick_labels.append(f"{pd.Timestamp(2000, month, 1).strftime('%b')} [{year_code}]")
    month_order = np.argsort(month_tick_positions)
    month_tick_positions = [month_tick_positions[index] for index in month_order]
    month_tick_labels = [month_tick_labels[index] for index in month_order]
    if resolution_info.name == "daily":
        day_slots = np.arange(1, 367, 30, dtype=int)
        day_tick_positions = np.concatenate(
            (
                [float(np.where(one_cycle_order == slot - 1)[0][0]) for slot in day_slots],
                [float(cycle_length + np.where(one_cycle_order == slot - 1)[0][0]) for slot in day_slots],
            )
        )
        day_tick_labels = tuple(str(int(slot)) for slot in np.tile(day_slots, 2))
    else:
        day_tick_positions = np.array([], dtype=float)
        day_tick_labels = ()
    target = {}
    other = {}
    valid_fraction = {}
    target_set = {int(year) for year in target_years}

    def ordered_year(year: int) -> tuple[np.ndarray, np.ndarray]:
        grouped = frame.loc[frame["year"] == year].groupby("slot")["value"].mean()
        counts = frame.loc[frame["year"] == year].groupby("slot")["value"].count()
        values = grouped.reindex(slot_values).to_numpy()[one_cycle_order]
        valid = counts.reindex(slot_values, fill_value=0).to_numpy()[one_cycle_order] > 0
        return values, valid

    for year in sorted(target_set):
        previous_values, previous_valid = ordered_year(year - 1)
        current_values, current_valid = ordered_year(year)
        target[year] = np.concatenate((previous_values, current_values))
        valid_fraction[year] = np.concatenate((previous_valid, current_valid))
    if other_years:
        for year in range(other_years[0], other_years[1] + 1):
            if year in target_set:
                continue
            previous_values, _ = ordered_year(year - 1)
            current_values, _ = ordered_year(year)
            other[year] = np.concatenate((previous_values, current_values))
    baseline = frame
    if baseline_years:
        baseline = baseline.loc[baseline["year"].between(*baseline_years)]
    climatology_one_cycle = (
        baseline.groupby("slot")["value"].mean().reindex(slot_values).to_numpy()[one_cycle_order]
    )
    climatology = np.tile(climatology_one_cycle, 2)
    warnings = []
    if frame["value"].notna().sum() == 0:
        warnings.append("The polygon contains no finite values.")
    if any(not valid[0] for valid in valid_fraction.values()):
        warnings.append("One or more target years have no complete previous-year continuity window.")
    return AnnualCycleResult(
        resolution_info.name,
        x,
        target,
        other,
        climatology,
        tuple(month_tick_labels),
        valid_fraction,
        tuple(warnings),
        np.asarray(month_tick_positions, dtype=float),
        tuple(month_tick_labels),
        day_tick_positions,
        day_tick_labels,
        float(cycle_length),
    )
