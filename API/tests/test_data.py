import numpy as np
from maize import config as C
from maize.soil import load_soil_grid, soil_lookup
from maize.transformers import Winsorizer, build_preprocessor
import pandas as pd


def test_soil_lookup_inside_and_outside():
    g = load_soil_grid(C.SOIL_GRID)
    s = soil_lookup(g, [0.28, 60.0], [34.75, 34.75])
    assert s.loc[0, "soil_ph"] > 3 and np.isnan(s.loc[1, "soil_ph"])


def test_winsorizer_learns_on_train_only():
    w = Winsorizer(1, 99).fit(np.arange(100.0).reshape(-1, 1))
    assert w.transform([[1000.0]])[0, 0] <= 99


def test_preprocessor_handles_blanks():
    X = pd.DataFrame({"a": [1.0, 2, 3, 100], "b": [0, 1, 1, 0], "c": ["x", "y", "x", "y"]})
    p = build_preprocessor(["a"], [], ["b"], ["c"]).fit(X)
    out = p.transform(pd.DataFrame({"a": [np.nan], "b": [np.nan], "c": [np.nan]}))
    assert np.isfinite(out).all()
