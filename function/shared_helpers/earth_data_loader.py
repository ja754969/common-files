"""
earth_data_loader.py
====================

Universal, lazy, zero-local-disk remote dataset loader for Earth-science dashboards.

Single public entry point
-------------------------
    get_remote_dataset(resource, provider, data_type, username, password,
                       chunks, gee_project, gee_grid) -> xarray.Dataset

Supported infrastructure
------------------------
    data_type="opendap"  public THREDDS (NOAA NCEI/PSL) and Earthdata-protected
                         Hyrax/THREDDS (NASA) through xarray + pydap / netCDF4.
    data_type="cog"      Cloud-Optimized GeoTIFFs through rioxarray / rasterio
                         (GDAL /vsicurl HTTP range requests).
    data_type="gee"      Google Earth Engine Image / ImageCollection through Xee.

Design contract
---------------
1. The returned Dataset is lazy. Only metadata and 1-D index coordinates are
   fetched; every data variable is a Dask graph (or Xee lazy array). Calling the
   function on every Streamlit rerun therefore costs one metadata handshake.
2. The function is decoupled from variables, time windows and bounding boxes.
   Subsetting is the caller's job (``.sel`` / ``.isel`` / ``.rio.clip_box``).
3. Axis names are normalised to ``lat`` / ``lon`` / ``time`` and the grid
   topology is advertised in ``ds.attrs["grid_topology"]``.
4. Network state (HTTP session pools, GDAL options, the Earth Engine
   initialisation) lives in module-level registries guarded by locks, so the
   function is safe to call from many Streamlit script threads and safe to wrap
   in ``@st.cache_resource``.

Credentials are never accepted from the manifest. They are read from the
``username`` / ``password`` arguments or, by default, from the environment
variables, or a local ignored credentials file.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

__all__ = [
    "get_remote_dataset",
    "get_remote_dataset_from_config",
    "load_manifest",
    "configure_dask_for_streaming",
    "authenticate_earth_engine",
    "save_earthdata_credentials",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

ENV_EARTHDATA_USER = "EARTHDATA_USERNAME"
ENV_EARTHDATA_PASS = "EARTHDATA_PASSWORD"
ENV_EE_SERVICE_ACCOUNT_JSON = "EE_SERVICE_ACCOUNT_JSON"  # optional, headless servers
ENV_MANIFEST_PATH = "EARTH_DATA_CONFIG"
ENV_CREDENTIALS_PATH = "EARTH_DATA_CREDENTIALS"
DEFAULT_CREDENTIALS_PATH = Path.home() / ".earth-data-detective" / "credentials.json"

# Providers whose endpoints sit behind NASA Earthdata Login (URS).
_AUTH_PROVIDERS = {"nasa", "earthdata", "nasa_earthdata", "podaac", "lpdaac", "gesdisc"}

_VALID_DATA_TYPES = {"opendap", "cog", "gee"}

# Engine preference for OPeNDAP. netCDF4 (netCDF-C) is the most battle-tested DAP
# client for THREDDS, but needs a DAP-enabled build and serialises reads behind
# the HDF5 lock. pydap is pure Python, accepts an authenticated requests.Session
# and is the only engine used for URS-protected NASA endpoints.
_PUBLIC_OPENDAP_ENGINES: Tuple[str, ...] = ("netcdf4", "pydap")
_AUTH_OPENDAP_ENGINES: Tuple[str, ...] = ("pydap",)

# Axis alias table (all lower-case). Order = priority. Ambiguous short aliases
# ("x", "y", "t") sit last and are only honoured when they are coordinates or
# dimensions (and, for x/y, when their values look geographic).
_AXIS_ALIASES: Dict[str, Tuple[str, ...]] = {
    "lat": ("lat", "latitude", "lats", "latitudes", "nav_lat", "ylat", "lat_rho", "y"),
    "lon": ("lon", "longitude", "lons", "longitudes", "nav_lon", "xlon", "long", "lon_rho", "x"),
    "time": ("time", "times", "date", "datetime", "valid_time", "time_counter", "ocean_time", "t"),
}
_AMBIGUOUS_ALIASES = {"x", "y", "t"}

# Variables / attributes that identify an unstructured mesh (FVCOM, UGRID, ...).
_MESH_CONNECTIVITY_VARS = {"nv", "face_nodes", "face_node_connectivity", "triangles"}

# GDAL tuning for HTTP range-request streaming. Small caches keep memory minimal.
_GDAL_BASE_OPTIONS: Dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",  # never list the remote "directory"
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_INGESTED_BYTES_AT_OPEN": "32768",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(16 * 1024 * 1024),
}
# Tiny cookie jar (a few hundred bytes, no data) needed by the URS redirect flow.
_GDAL_COOKIE_PATH = os.path.join(tempfile.gettempdir(), "earthdata_gdal_cookies.txt")

_EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"

# --------------------------------------------------------------------------- #
# Process-wide registries (thread-safe, reusable across Streamlit reruns)     #
# --------------------------------------------------------------------------- #

_SESSION_LOCK = threading.Lock()
_SESSION_REGISTRY: Dict[str, requests.Session] = {}

_EE_LOCK = threading.Lock()
_EE_INITIALISED_PROJECT: Optional[str] = None


# --------------------------------------------------------------------------- #
# Small utilities                                                             #
# --------------------------------------------------------------------------- #

def _require(module_name: str, install_hint: str):
    """Import an optional heavy dependency lazily with an actionable error."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"'{module_name}' is required for this data_type. Install it with: pip install {install_hint}"
        ) from exc


def configure_dask_for_streaming(num_workers: int = 16, chunk_size: str = "64MiB") -> None:
    """
    Optional one-off tuning for network-bound workloads. Call once at app start.

    The default Dask thread pool is sized to the CPU count, which under-uses the
    network; HTTP range requests are I/O-bound so more threads help. A modest
    ``array.chunk-size`` keeps peak memory low and keeps each request small.
    This mutates global Dask configuration, hence it is never called implicitly.
    """
    import dask

    dask.config.set(scheduler="threads", num_workers=num_workers, **{"array.chunk-size": chunk_size})


def _resolve_credentials(username: Optional[str], password: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve credentials without reading them from the dataset manifest."""
    if not (username and password):
        configured_path = os.environ.get(ENV_CREDENTIALS_PATH)
        candidate_paths = (
            (Path(configured_path),)
            if configured_path
            else (
                Path.cwd() / "config" / "earthdata_credentials.json",
                DEFAULT_CREDENTIALS_PATH,
            )
        )
        credentials_path = next((path for path in candidate_paths if path.is_file()), None)
        if credentials_path is not None:
            try:
                with credentials_path.open("r", encoding="utf-8") as handle:
                    stored = json.load(handle)
            except (OSError, ValueError) as exc:
                raise PermissionError(f"Unable to read credentials file {credentials_path}: {exc}") from exc
            username = username or stored.get("username")
            password = password or stored.get("password")
    return (
        username or os.environ.get(ENV_EARTHDATA_USER),
        password or os.environ.get(ENV_EARTHDATA_PASS),
    )


def _is_auth_provider(provider: str) -> bool:
    return (provider or "").strip().lower() in _AUTH_PROVIDERS


# --------------------------------------------------------------------------- #
# HTTP session pool                                                           #
# --------------------------------------------------------------------------- #

class EarthdataSession(requests.Session):
    """
    ``requests.Session`` that keeps Basic-Auth credentials across the redirect
    hops of NASA Earthdata Login (data host -> urs.earthdata.nasa.gov -> data host)
    but strips them if a redirect leaves to any unrelated host.

    Defined at module level and holding only plain data, so instances remain
    picklable (requests.Session implements __getstate__/__setstate__).
    """

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username: str, password: str):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):  # noqa: D401 - requests hook
        headers = prepared_request.headers
        if "Authorization" in headers:
            original_host = requests.utils.urlparse(response.request.url).hostname
            redirect_host = requests.utils.urlparse(prepared_request.url).hostname
            if (
                original_host != redirect_host
                and redirect_host != self.AUTH_HOST
                and original_host != self.AUTH_HOST
            ):
                del headers["Authorization"]


def _mount_connection_pool(session: requests.Session) -> requests.Session:
    """Attach a pooled, retrying adapter so parallel Dask threads reuse sockets."""
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
    )
    adapter = HTTPAdapter(pool_connections=16, pool_maxsize=64, max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_public_session() -> requests.Session:
    """Shared anonymous session for public THREDDS servers."""
    with _SESSION_LOCK:
        session = _SESSION_REGISTRY.get("public")
        if session is None:
            session = _mount_connection_pool(requests.Session())
            _SESSION_REGISTRY["public"] = session
        return session


def _get_earthdata_session(username: str, password: str) -> requests.Session:
    """
    Persistent authenticated session, one per credential pair. The registry key is a
    digest, so the plain password is not retained as a dictionary key.
    """
    key = "urs:" + hashlib.sha256(f"{username}\0{password}".encode()).hexdigest()
    with _SESSION_LOCK:
        session = _SESSION_REGISTRY.get(key)
        if session is None:
            session = _mount_connection_pool(EarthdataSession(username, password))
            _SESSION_REGISTRY[key] = session
        return session


# --------------------------------------------------------------------------- #
# Backend openers (each returns a *lazy* xr.Dataset)                          #
# --------------------------------------------------------------------------- #

def _open_opendap(url: str, provider: str, username, password, chunks) -> xr.Dataset:
    """OPeNDAP byte-range streaming through xarray (pydap / netCDF4 backends)."""
    if _is_auth_provider(provider):
        user, pwd = _resolve_credentials(username, password)
        if not (user and pwd):
            raise PermissionError(
                f"Provider '{provider}' requires Earthdata credentials. Set {ENV_EARTHDATA_USER} and "
                f"{ENV_EARTHDATA_PASS} or pass username/password."
            )
        session = _get_earthdata_session(user, pwd)
        engines = _AUTH_OPENDAP_ENGINES
    else:
        session = _get_public_session()
        engines = _PUBLIC_OPENDAP_ENGINES

    failures: List[str] = []
    for engine in engines:
        kwargs: Dict[str, Any] = {"engine": engine, "chunks": chunks}
        if engine == "pydap":
            kwargs["session"] = session  # pooled (and, for NASA, authenticated) transport
        try:
            # Passing `chunks` makes xarray build a Dask graph instead of reading arrays.
            return xr.open_dataset(url, **kwargs)
        except Exception as exc:  # noqa: BLE001 - we fall through to the next engine
            failures.append(f"{engine}: {exc!r}")
            logger.debug("OPeNDAP engine %s failed for %s", engine, url, exc_info=True)

    raise ConnectionError(f"Could not open OPeNDAP resource {url!r}. Attempts -> " + " | ".join(failures))


def _publish_gdal_options(options: Dict[str, str]) -> None:
    """
    Make GDAL options visible to Dask worker threads.

    Dask performs the actual range reads later, in pool threads, long after any
    ``with rasterio.Env(...)`` block has exited, and rasterio.Env is thread-local.
    GDAL falls back to process environment variables for configuration, so mirroring
    the options there guarantees every read is authenticated and tuned.

    Security note: the environment is process-wide. Use a dedicated service account.
    """
    for key, value in options.items():
        os.environ[key] = value


def _open_cog(url: str, provider: str, username, password, chunks) -> xr.Dataset:
    """Cloud-Optimized GeoTIFF streaming (HTTP range requests) through rioxarray."""
    rioxarray = _require("rioxarray", "rioxarray")
    rasterio = _require("rasterio", "rasterio")

    options = dict(_GDAL_BASE_OPTIONS)
    if _is_auth_provider(provider):
        user, pwd = _resolve_credentials(username, password)
        if not (user and pwd):
            raise PermissionError(
                f"Provider '{provider}' requires Earthdata credentials. Set {ENV_EARTHDATA_USER} and "
                f"{ENV_EARTHDATA_PASS} or pass username/password."
            )
        options.update(
            GDAL_HTTP_USERPWD=f"{user}:{pwd}",
            GDAL_HTTP_COOKIEFILE=_GDAL_COOKIE_PATH,
            GDAL_HTTP_COOKIEJAR=_GDAL_COOKIE_PATH,
        )
    _publish_gdal_options(options)

    # The Env block covers the handshake (header/IFD reads); _publish_gdal_options
    # covers the deferred chunk reads that happen later in Dask threads.
    with rasterio.Env(**options):
        obj = rioxarray.open_rasterio(url, chunks=chunks, masked=True)

    if isinstance(obj, xr.DataArray):
        ds = obj.to_dataset(name=obj.name or "band_data")
    elif isinstance(obj, xr.Dataset):
        ds = obj
    else:  # list of per-subdataset objects
        ds = xr.merge([o.to_dataset(name=o.name or f"band_data_{i}") if isinstance(o, xr.DataArray) else o
                       for i, o in enumerate(obj)])

    crs = ds.rio.crs
    if crs is not None:
        ds.attrs["source_crs"] = crs.to_string()
        ds.attrs["_source_crs_is_geographic"] = int(bool(crs.is_geographic))
    return ds


def _ee_credentials(ee):
    """Service-account credentials for headless servers; None means 'use persistent/ADC'."""
    raw = os.environ.get(ENV_EE_SERVICE_ACCOUNT_JSON)
    if not raw:
        return None
    info = json.loads(raw)
    return ee.ServiceAccountCredentials(info["client_email"], key_data=raw)


def authenticate_earth_engine(project: str) -> None:
    """Run the interactive Earth Engine OAuth flow from a terminal."""
    project = project.strip()
    if not project:
        raise ValueError("A Google Cloud project ID is required for Earth Engine authentication.")
    ee = _require("ee", "earthengine-api")
    ee.Authenticate()
    ee.Initialize(project=project, opt_url=_EE_HIGH_VOLUME_URL)


def save_earthdata_credentials(
    username: str,
    password: str,
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """Persist Earthdata credentials in a private local JSON file."""
    username = username.strip()
    if not username or not password:
        raise ValueError("Both an Earthdata username and password are required.")
    destination = Path(path) if path is not None else DEFAULT_CREDENTIALS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps({"username": username, "password": password}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except OSError as exc:
        raise PermissionError(f"Unable to save Earthdata credentials to {destination}: {exc}") from exc
    return destination


def _ensure_ee_initialised(ee, project: Optional[str]) -> None:
    """Initialise Earth Engine exactly once per project per process (idempotent, locked)."""
    global _EE_INITIALISED_PROJECT
    if not project:
        raise ValueError("gee_project (a Google Cloud project ID) is required for data_type='gee'.")
    with _EE_LOCK:
        if _EE_INITIALISED_PROJECT == project:
            return
        kwargs: Dict[str, Any] = {"project": project, "opt_url": _EE_HIGH_VOLUME_URL}
        creds = _ee_credentials(ee)
        if creds is not None:
            kwargs["credentials"] = creds
        try:
            ee.Initialize(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise PermissionError(
                "Earth Engine initialisation failed. On a server, supply a service account via "
                f"{ENV_EE_SERVICE_ACCOUNT_JSON} (or application-default credentials); "
                "interactive ee.Authenticate() is not possible inside a web app."
            ) from exc
        _EE_INITIALISED_PROJECT = project


def _to_ee_geometry(ee, geometry):
    """Accept an ee.Geometry, a GeoJSON dict, or a [west, south, east, north] bbox."""
    if isinstance(geometry, ee.Geometry):
        return geometry
    if isinstance(geometry, dict):
        return ee.Geometry(geometry)
    if isinstance(geometry, (list, tuple)) and len(geometry) == 4:
        return ee.Geometry.Rectangle(list(geometry), proj="EPSG:4326", geodesic=False)
    raise TypeError("gee_grid['geometry'] must be an ee.Geometry, a GeoJSON dict or [w, s, e, n].")


def _resolve_ee_object(ee, resource):
    """Turn an asset ID string into an ee.Image / ee.ImageCollection (metadata call only)."""
    if isinstance(resource, ee.ComputedObject):
        return resource  # caller already built (and filtered) the object
    asset_type = ee.data.getAsset(resource).get("type")
    if asset_type == "IMAGE_COLLECTION":
        return ee.ImageCollection(resource)
    if asset_type == "IMAGE":
        return ee.Image(resource)
    raise ValueError(f"Earth Engine asset {resource!r} has unsupported type {asset_type!r}.")


def _open_gee(resource, gee_project, gee_grid, chunks) -> xr.Dataset:
    """Bind an Earth Engine Image / ImageCollection to a lazy Dataset through Xee."""
    ee = _require("ee", "earthengine-api")
    _require("xee", "xee")  # importing registers the engine="ee" backend

    _ensure_ee_initialised(ee, gee_project)

    grid = dict(gee_grid or {})
    if "geometry" in grid and grid["geometry"] is not None:
        grid["geometry"] = _to_ee_geometry(ee, grid["geometry"])
    grid.setdefault("crs", "EPSG:4326")
    if "scale" not in grid and "crs_transform" not in grid:
        raise ValueError("gee_grid must define 'scale' (CRS units per pixel) or 'crs_transform' + 'shape_2d'.")
    grid.setdefault("ee_initialize", False)  # we already initialised with the explicit project

    # Dask's 'auto' targets ~128 MiB chunks, which would exceed Earth Engine's
    # per-request payload cap. {} tells xarray to use Xee's own request-sized chunks.
    xee_chunks = {} if chunks == "auto" else chunks

    ee_obj = _resolve_ee_object(ee, resource)
    return xr.open_dataset(ee_obj, engine="ee", chunks=xee_chunks, **grid)


# --------------------------------------------------------------------------- #
# Axis standardisation interceptor                                            #
# --------------------------------------------------------------------------- #

def _is_geographic_axis(ds: xr.Dataset, name: str, canonical: str, known_geographic: Optional[bool]) -> bool:
    """
    Decide whether an ambiguous axis ('x'/'y') really holds lon/lat degrees. Projected
    grids (UTM metres, polar stereographic) must NOT be renamed to lon/lat.
    Only a small 1-D index coordinate is inspected, never data variables.
    """
    if known_geographic is not None:
        return bool(known_geographic)
    var = ds.variables.get(name)
    if var is None or var.ndim != 1 or not np.issubdtype(var.dtype, np.number):
        return False
    values = np.asarray(var.values, dtype="float64")
    if values.size == 0 or not np.isfinite(values).any():
        return False
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    return (-90.0 <= lo and hi <= 90.0) if canonical == "lat" else (-180.0 <= lo and hi <= 360.0)


def _standardize_axes(ds: xr.Dataset, known_geographic: Optional[bool] = None) -> xr.Dataset:
    """
    Fuzzy-map axis naming variants onto the canonical 'lat' / 'lon' / 'time'.

    Handles dimension coordinates, dimensions without coordinates, and
    unstructured-mesh cases where lat/lon are plain data variables on a shared
    'node' dimension (those are promoted to coordinates afterwards).
    """
    lowered: Dict[str, Any] = {}
    for original in list(ds.variables) + list(ds.dims):
        lowered.setdefault(str(original).lower().strip(), original)
    coord_or_dim = set(ds.coords) | set(ds.dims)

    renames: Dict[Any, str] = {}
    for canonical, aliases in _AXIS_ALIASES.items():
        if canonical in ds.variables or canonical in ds.dims:
            continue  # already canonical; never clobber
        for alias in aliases:
            original = lowered.get(alias)
            if original is None or original == canonical or original in renames:
                continue
            if alias in _AMBIGUOUS_ALIASES:
                if original not in coord_or_dim:
                    continue
                if canonical in ("lat", "lon") and not _is_geographic_axis(ds, original, canonical, known_geographic):
                    continue
            renames[original] = canonical
            break

    if renames:
        ds = ds.rename(renames)
        ds.attrs["axis_renames"] = ", ".join(f"{old}->{new}" for old, new in renames.items())

    promote = [n for n in ("lat", "lon", "time") if n in ds.variables and n not in ds.coords]
    if promote:
        ds = ds.set_coords(promote)
    return ds


# --------------------------------------------------------------------------- #
# Grid topology classification                                                #
# --------------------------------------------------------------------------- #

def _classify_grid_topology(ds: xr.Dataset) -> Tuple[str, str]:
    """
    Return (topology, reason) where topology is 'regular', 'unstructured' or 'unknown'.

    'unstructured' covers every case where `.sel(lat=..., lon=...)` is NOT valid:
    curvilinear 2-D lat/lon, 1-D lat/lon sharing a node/point dimension (triangular
    meshes such as FVCOM, point sets, tracks), and datasets carrying mesh
    connectivity metadata (UGRID cf_role / 'nv' style variables).
    """
    variables = ds.variables

    has_mesh_metadata = any(
        v.attrs.get("cf_role") == "mesh_topology" or "topology_dimension" in v.attrs for v in variables.values()
    ) or any(name in variables for name in _MESH_CONNECTIVITY_VARS)
    if has_mesh_metadata:
        return "unstructured", "mesh connectivity metadata present"

    lat, lon = variables.get("lat"), variables.get("lon")
    if lat is not None and lon is not None:
        if lat.ndim >= 2 or lon.ndim >= 2:
            return "unstructured", "curvilinear: multi-dimensional lat/lon coordinates"
        if lat.ndim == 1 and lon.ndim == 1:
            if lat.dims == lon.dims:
                return "unstructured", f"lat/lon share the single dimension {lat.dims[0]!r}"
            return "regular", "orthogonal 1-D lat/lon axes"
        return "unknown", "scalar lat/lon"

    x, y = variables.get("x"), variables.get("y")
    if x is not None and y is not None and x.ndim == 1 and y.ndim == 1 and x.dims != y.dims:
        return "regular", "orthogonal projected x/y axes (not lat/lon degrees)"
    return "unknown", "no recognisable horizontal coordinates"


# --------------------------------------------------------------------------- #
# Laziness guard, aggregation and provenance                                  #
# --------------------------------------------------------------------------- #

def _ensure_lazy(ds: xr.Dataset) -> xr.Dataset:
    """
    Safety net: every non-scalar data variable must be Dask-backed. Re-chunking a
    lazily indexed backend array builds a graph and reads nothing.
    """
    eager = [name for name, var in ds.data_vars.items() if var.ndim > 0 and var.chunks is None]
    if eager:
        logger.warning("Variables %s were not Dask-backed; wrapping them lazily.", eager)
        ds = ds.chunk("auto")
    return ds


def _aggregate_along_time(datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    """
    Stitch independent lazy datasets along 'time' without touching data.

    The concat options are chosen so no array is ever loaded to compare it:
      data_vars/coords='minimal' - only variables already carrying 'time' are joined
      compat='override'          - time-invariant variables are taken from the first file
      join='exact'               - horizontal grids must match, otherwise fail loudly
    """
    prepared: List[xr.Dataset] = []
    for position, ds in enumerate(datasets):
        if "time" not in ds.dims:
            if "time" in ds.coords and ds["time"].ndim == 0:
                ds = ds.expand_dims("time")
            else:
                raise ValueError(
                    f"Resource #{position} has no 'time' axis, so it cannot be concatenated along time. "
                    "Pass a single resource, or only resources that carry a time coordinate."
                )
        prepared.append(ds)

    merged = xr.concat(
        prepared,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
        combine_attrs="override",
    )

    time_index = merged.indexes["time"]
    if not time_index.is_monotonic_increasing:
        merged = merged.sortby("time")  # lazy re-indexing of the Dask graph
    if not time_index.is_unique:
        logger.warning("Concatenated time axis contains duplicate timestamps (overlapping inputs?).")
    return merged


def _finalize(ds: xr.Dataset, resource, provider: str, data_type: str) -> xr.Dataset:
    """Classify topology and stamp provenance attributes on a freshly built Dataset."""
    topology, reason = _classify_grid_topology(ds)
    ds.attrs["grid_topology"] = topology
    ds.attrs["grid_topology_reason"] = reason
    if isinstance(resource, str):
        ds.attrs["source_resource"] = resource
    elif isinstance(resource, (list, tuple)):
        ds.attrs["source_resource"] = json.dumps([str(r) for r in resource])
    else:  # e.g. a pre-built ee.ImageCollection
        ds.attrs["source_resource"] = f"<{type(resource).__name__}>"
    ds.attrs["source_provider"] = provider
    ds.attrs["source_data_type"] = data_type
    ds.attrs.pop("_source_crs_is_geographic", None)
    return ds


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def get_remote_dataset(
    resource,
    provider="generic",
    data_type="opendap",
    username=None,
    secret=None,
    chunks="auto",
    gee_project=None,
    gee_grid=None,
) -> xr.Dataset:
    """
    Open any supported remote Earth-science resource as a purely lazy ``xr.Dataset``.

    Parameters
    ----------
    resource : str | Sequence[str] | ee.Image | ee.ImageCollection
        One URL / asset ID, or a list of them. A list is opened recursively, one
        independent lazy connection per element, and stitched along ``time``.
        For 'gee' an already-filtered ee object may be passed instead of an ID.
    provider : str
        'generic' / 'noaa' for public servers; 'nasa' (or 'earthdata', 'podaac',
        'lpdaac', 'gesdisc') to route through Earthdata Login authentication.
    data_type : {'opendap', 'cog', 'gee'}
        Transport/backend selector.
    username, password : str, optional
        Earthdata credentials. Default: EARTHDATA_USERNAME / EARTHDATA_PASSWORD.
    chunks : 'auto' | dict | int
        Dask chunking. Dict keys refer to the SOURCE dimension names (before
        standardisation), e.g. {'time': 12, 'latitude': 360}. For 'gee', 'auto'
        means Xee's request-sized chunks.
    gee_project : str
        Google Cloud project ID used for ``ee.Initialize`` (data_type='gee').
    gee_grid : dict
        Xee grid definition: crs, scale (or crs_transform + shape_2d) and an
        optional geometry (ee.Geometry, GeoJSON dict or [w, s, e, n]).

    Returns
    -------
    xr.Dataset
        Lazy dataset with axes renamed to lat/lon/time where possible and
        ``attrs['grid_topology']`` in {'regular', 'unstructured', 'unknown'}.

    Notes
    -----
    The returned object is shared when wrapped in ``st.cache_resource``; treat it as
    read-only (derive new objects with ``.sel`` / ``.isel``; never assign into
    ``ds.attrs`` or ``ds[...]`` after it leaves this function).
    """
    kind = (data_type or "").strip().lower()
    if kind not in _VALID_DATA_TYPES:
        raise ValueError(f"data_type must be one of {sorted(_VALID_DATA_TYPES)}, got {data_type!r}.")
    if chunks is None or chunks is False:
        chunks = "auto"  # strict-laziness invariant: a Dask graph is always requested

    # ---- Multi-resource aggregation (recursive) --------------------------------
    if isinstance(resource, (list, tuple)):
        if not resource:
            raise ValueError("'resource' is an empty list.")
        parts = [
            get_remote_dataset(item, provider, kind, username, secret, chunks, gee_project, gee_grid)
            for item in resource
        ]
        merged = parts[0] if len(parts) == 1 else _aggregate_along_time(parts)
        return _finalize(merged, resource, provider, kind)

    # ---- Single resource -------------------------------------------------------
    known_geographic: Optional[bool] = None
    if kind == "opendap":
        ds = _open_opendap(resource, provider, username, secret, chunks)
    elif kind == "cog":
        ds = _open_cog(resource, provider, username, secret, chunks)
        flag = ds.attrs.get("_source_crs_is_geographic")
        known_geographic = None if flag is None else bool(flag)
    else:  # 'gee'
        ds = _open_gee(resource, gee_project, gee_grid, chunks)

    ds = _standardize_axes(ds, known_geographic=known_geographic)
    ds = _ensure_lazy(ds)
    return _finalize(ds, resource, provider, kind)


# --------------------------------------------------------------------------- #
# Manifest adapter (optional convenience; get_remote_dataset stays decoupled) #
# --------------------------------------------------------------------------- #

def load_manifest(path: Optional[str] = None) -> Dict[str, Any]:
    """Read the JSON manifest (default: $EARTH_DATA_CONFIG or data_config.json beside this file)."""
    path = path or os.environ.get(ENV_MANIFEST_PATH) or os.path.join(os.path.dirname(__file__), "data_config.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_remote_dataset_from_config(dataset_key: str, config_path: Optional[str] = None) -> xr.Dataset:
    """
    Resolve a manifest entry and delegate to ``get_remote_dataset``.

    An entry supplies either ``resource`` (string or list) or ``resource_template``
    with ``years: [first, last]`` (inclusive), which expands to one URL per year.
    Credentials are deliberately NOT read from the manifest.
    """
    manifest = load_manifest(config_path)
    try:
        entry = {**manifest.get("defaults", {}), **manifest["datasets"][dataset_key]}
    except KeyError as exc:
        raise KeyError(f"Dataset key {dataset_key!r} not found in manifest.") from exc

    if "resource_template" in entry:
        first, last = entry["years"]
        resource: Union[str, List[str]] = [entry["resource_template"].format(year=y) for y in range(first, last + 1)]
    else:
        resource = entry["resource"]

    return get_remote_dataset(
        resource,
        provider=entry.get("provider", "generic"),
        data_type=entry.get("data_type", "opendap"),
        chunks=entry.get("chunks", "auto"),
        gee_project=entry.get("gee_project") or os.environ.get("GEE_PROJECT"),
        gee_grid=entry.get("gee_grid"),
    )
