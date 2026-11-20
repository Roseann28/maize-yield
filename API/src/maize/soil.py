"""Soil lookup from the iSDAsoil derived grid (about 1 km, topsoil 0 to 20 cm)."""
import numpy as np
import pandas as pd
from .config import SOIL


def load_soil_grid(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def soil_lookup(grid, lat, lon):
    """Soil values at lat/lon. NaN outside the grid or on a masked cell (water, no data)."""
    lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
    step = float(grid["step"])
    r = np.floor((float(grid["north"]) - lat) / step)
    c = np.floor((lon - float(grid["west"])) / step)
    nrow, ncol = grid["soil_ph"].shape
    ok = np.isfinite(r) & np.isfinite(c) & (r >= 0) & (r < nrow) & (c >= 0) & (c < ncol)
    ri, ci = np.where(ok, r, 0).astype(int), np.where(ok, c, 0).astype(int)
    return pd.DataFrame({k: np.where(ok, grid[k][ri, ci].astype(float), np.nan) for k in SOIL})
