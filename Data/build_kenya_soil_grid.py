"""Builds the soil grid used by the notebook, the training script and the API.

Run it once, on a computer with internet:
    pip install rasterio numpy
    python Data/build_kenya_soil_grid.py

It reads the iSDAsoil maps for the topsoil (0 to 20 cm) and averages them onto a grid of about 1 km over Kenya
(0.01 degree cells) for pH, organic carbon, total nitrogen, extractable phosphorus, extractable potassium, sand and clay.
The result is saved as Data/kenya_soil_grid.npz, next to this file. The soil of a field is then looked up from its coordinates.

Check the layer URLs and the back transforms below against the iSDAsoil documentation (Zenodo) before relying on it.
"""
from pathlib import Path
import numpy as np, rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
BASE = "https://isdasoil.s3.amazonaws.com/soil_data"
W, S, E_, N = 33.9, -4.7, 41.9, 5.5          # Kenya bounding box
STEP = 0.01                                    # degrees, about 1.1 km
LAYERS = {
 "soil_ph":         ("ph/ph.tif",                                            lambda v: v / 10),
 "soil_carbon":     ("carbon_organic/carbon_organic.tif",                    lambda v: np.expm1(v / 10)),
 "soil_nitrogen":   ("nitrogen_total/nitrogen_total.tif",                    lambda v: np.expm1(v / 100)),
 "soil_phosphorus": ("phosphorous_extractable/phosphorous_extractable.tif",  lambda v: np.expm1(v / 10)),
 "soil_potassium":  ("potassium_extractable/potassium_extractable.tif",      lambda v: np.expm1(v / 10)),
 "soil_sand":       ("sand_content/sand_content.tif",                        lambda v: v),
 "soil_clay":       ("clay_content/clay_content.tif",                        lambda v: v),
}
ncol, nrow = round((E_ - W) / STEP), round((N - S) / STEP)
out = {}
for col, (path, f) in LAYERS.items():
    with rasterio.open(f"{BASE}/{path}") as r:     # band 1 = 0 to 20 cm; reads via overviews
        l, b, rt, t = transform_bounds('EPSG:4326', r.crs, W, S, E_, N)   # maps are in metres, not degrees
        win = from_bounds(l, b, rt, t, r.transform)
        a = r.read(1, window=win, out_shape=(nrow, ncol), resampling=Resampling.average, masked=True)
        out[col] = f(a.astype(float).filled(np.nan)).astype("float32")
    if not np.isfinite(out[col]).any(): raise SystemExit(f"{col}: no data read (crs={r.crs}); check the layer URL")
    print(col, "done", np.nanmin(out[col]), np.nanmax(out[col]))
OUT_FILE = Path(__file__).resolve().parent / "kenya_soil_grid.npz"     # saved next to this script, in the Data folder
np.savez_compressed(OUT_FILE, west=W, south=S, east=E_, north=N, step=STEP, **out)
print("saved", OUT_FILE)
