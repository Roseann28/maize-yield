"""Prediction: soil lookup + practices -> yield, reasons (SHAP), coverage check and a likely range."""
import json
import joblib, numpy as np, pandas as pd, shap
from sklearn.neighbors import BallTree
from . import config as C
from .soil import load_soil_grid, soil_lookup

COVERAGE_WARN_KM = 50.0


class Predictor:
    def __init__(self, model_file=C.MODEL_FILE, meta_file=C.METADATA_FILE, grid_file=C.SOIL_GRID):
        art = joblib.load(model_file)
        self.pipe, self.features, self.defaults = art["pipeline"], art["features"], art["defaults"]
        self.meta = json.loads(meta_file.read_text())
        self.grid = load_soil_grid(grid_file)
        self.prep, self.model = self.pipe.named_steps["prep"], self.pipe.named_steps["model"]
        self.names = list(self.prep.get_feature_names_out())
        self.fmap = [self._to_feature(n) for n in self.names]
        self.explainer = shap.TreeExplainer(self.model)
        loc = np.load(C.MODEL_DIR / "survey_locations.npz")["latlon"]
        self.tree = BallTree(np.radians(loc), metric="haversine")
        q = self.meta["district_blocked_estimate"]["residual_quantiles_kg_ha"]
        self.q10, self.q90 = q["q10"], q["q90"]

    def _to_feature(self, name):
        r = (name.split("__", 1)[1] if "__" in name else name).replace("missingindicator_", "")
        cands = [f for f in self.features if r == f or r.startswith(f + "_")]
        return max(cands, key=len) if cands else r

    def coverage_km(self, lat, lon):
        d, _ = self.tree.query(np.radians([[lat, lon]]), k=1)
        return float(d[0, 0] * 6371.0088)

    def predict(self, lat, lon, practices=None, n_reasons=3):
        soil = soil_lookup(self.grid, [lat], [lon])
        if soil.isna().all(axis=None):
            raise ValueError("No soil data at this location (outside Kenya's land area or a masked cell such as water).")
        given = {k: v for k, v in (practices or {}).items() if v is not None}
        unknown = sorted(set(given) - set(C.PRACTICE_SET))
        if unknown:
            raise ValueError(f"Unknown practice fields: {unknown}")
        row = {**self.defaults, **given, **soil.iloc[0].to_dict()}
        x = pd.DataFrame([row])[self.features]
        pred = float(self.pipe.predict(x)[0])
        xt = pd.DataFrame(self.prep.transform(x), columns=self.names)
        sv = pd.Series(self.explainer.shap_values(xt)[0], index=self.names).groupby(self.fmap).sum()
        sv = sv.reindex(sv.abs().sort_values(ascending=False).index).head(n_reasons)
        km = self.coverage_km(lat, lon)
        notes = []
        if km > COVERAGE_WARN_KM:
            notes.append(f"This location is {km:.0f} km from the nearest surveyed field. The model was trained on western Kenya, "
                         "so treat this as an extrapolation.")
        used_defaults = sorted(set(C.PRACTICE_SET) - set(given))
        if used_defaults:
            notes.append(f"{len(used_defaults)} practice fields were not supplied and were filled with the survey median or most common value.")
        return {"predicted_yield_kg_ha": round(pred),
                "likely_range_kg_ha": [max(0, round(pred + self.q10)), round(pred + self.q90)],
                "range_meaning": "10th to 90th percentile of the error seen when whole districts were held out",
                "soil_used": {k: round(float(v), 2) for k, v in soil.iloc[0].items()},
                "top_reasons": [{"feature": f, "effect_kg_ha": round(float(v))} for f, v in sv.items()],
                "distance_to_nearest_surveyed_field_km": round(km, 1),
                "notes": notes}
