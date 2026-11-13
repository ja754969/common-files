"""Lazy ensemble diagnostics for gridded climate fields."""

from __future__ import annotations

import numpy as np
import xarray as xr


def ensemble_agreement_fraction(members: xr.DataArray) -> xr.DataArray:
    """Return the fraction of valid members sharing the ensemble-mean sign."""
    if "ensemble" not in members.dims:
        raise ValueError("An ensemble dimension is required for consensus diagnostics.")
    mean = members.mean("ensemble", skipna=True)
    valid = members.notnull()
    member_sign = xr.where(members > 0, 1, xr.where(members < 0, -1, 0))
    mean_sign = xr.where(mean > 0, 1, xr.where(mean < 0, -1, 0))
    agreement = xr.where(valid, member_sign == mean_sign, np.nan)
    return agreement.mean("ensemble", skipna=True)


def ensemble_consensus(
    members: xr.DataArray,
    threshold: float = 0.8,
) -> xr.DataArray:
    """Return a mask where at least ``threshold`` of members share the MMM sign."""
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("The consensus threshold must be between 0.5 and 1.0.")
    if "ensemble" not in members.dims:
        raise ValueError("An ensemble dimension is required for consensus diagnostics.")
    mean = members.mean("ensemble", skipna=True)
    mean_sign = xr.where(mean > 0, 1, xr.where(mean < 0, -1, 0))
    fraction = ensemble_agreement_fraction(members)
    # A zero MMM has no directional signal, even if all members are exactly zero.
    return (fraction >= threshold).where(mean.notnull() & (mean_sign != 0), False)
