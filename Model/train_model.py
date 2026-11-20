"""Training pipeline: ingest -> clean -> split -> train -> evaluate -> save -> predict.

Run it from the project folder:
    python Model/train_model.py

It needs Data/one_acre_fund.xlsx and Data/kenya_soil_grid.npz.
It follows the same steps as Notebook/maize_yield_analysis.ipynb and writes the files the API loads:
    Model/maize_model.joblib, Model/metadata.json, Model/survey_locations.npz
Training uses 2016 to 2019. The 2020 season is scored once, after training.
"""
import sys, json, platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd, joblib, sklearn, xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "API" / "src"))
# the API loads the saved model, so the model has to be built from the same preprocessing code the API has
from maize import __version__
from maize.soil import load_soil_grid, soil_lookup
from maize.transformers import build_preprocessor, choose_scalers

SURVEY = ROOT / "Data" / "one_acre_fund.xlsx"
SOIL_GRID = ROOT / "Data" / "kenya_soil_grid.npz"
MODEL_DIR = ROOT / "Model"

SEED = 42
ACRE_PER_HA = 2.471
SOIL = ["soil_ph", "soil_carbon", "soil_nitrogen", "soil_phosphorus", "soil_potassium", "soil_sand", "soil_clay"]
CONT = ["plant_doy", "plant_spacing", "row_spacing", "weed", "dap_kg_ha", "can_kg_ha", "urea_kg_ha", "npk_kg_ha", "lime_kg_ha",
        "plot_hectares", "hh_num", "hh_num_under18", "livestock"]
BINARY = ["intercrop", "compost", "pesticide", "pest_disease", "drought", "flood", "electricity"]
CAT = ["seed_type", "slope", "distance_meter", "fertility"]
PRACTICE_SET = CONT + BINARY + CAT
FEATURES = PRACTICE_SET + SOIL            # practices plus soil, the main model


# 1. Ingest: read the survey and add soil from the grid by field coordinates
print("1. Ingest")
raw = pd.read_excel(SURVEY)
grid = load_soil_grid(SOIL_GRID)
raw = pd.concat([raw.reset_index(drop=True), soil_lookup(grid, raw["field_latitude"], raw["field_longitude"])], axis=1)
print("   survey records:", len(raw))


# 2. Clean
print("2. Clean")
# same district spelt in different ways
raw["district"] = (raw["district"].str.lower().str.replace(r"[()\s]", "", regex=True)
                   .replace({"kakamegabnorth": "kakameganorth"}))
# blank districts take the district of the five nearest labelled fields (distance in km, labels only, no yield)
labelled = raw[raw.field_latitude.notna() & raw.district.notna()]
blank = raw.district.isna() & raw.field_latitude.notna()
knn = KNeighborsClassifier(5, metric="haversine").fit(np.radians(labelled[["field_latitude", "field_longitude"]].values),
                                                      labelled.district.astype(str).values)
raw.loc[blank, "district"] = knn.predict(np.radians(raw.loc[blank, ["field_latitude", "field_longitude"]].values))
print("   blank districts filled:", int(blank.sum()))

df = raw[raw["yield_kg_ph"].notna() & (raw["yield_kg_ph"] > 0)]
df = df[df["district"].notna() & df["field_latitude"].notna() & df["field_longitude"].notna() & df["soil_ph"].notna()].copy()
df["yield_kg_ha"] = df["yield_kg_ph"]
for f in ["dap", "can", "urea", "npk", "lime"]:                       # kg per acre to kg per hectare, above 2000 is an entry error
    v = df[f"{f}_kg_pa"] * ACRE_PER_HA
    df[f"{f}_kg_ha"] = v.where(v <= 2000)
df["plot_hectares"] = df["plot_acres"].replace(0, np.nan) / ACRE_PER_HA
df["weed"] = df["weed"].where(df["weed"] <= 6)                         # weeding more than 6 times is an entry error
df["plant_doy"] = df["plant_date"].dt.dayofyear.where(df["plant_date"].dt.month.between(1, 6))
df["livestock"] = df[["cows", "goats", "chickens"]].sum(axis=1, min_count=1)
df["seed_type"] = df["seed_type"].str.lower()
full = df[["district", "year", "yield_kg_ha", "field_latitude", "field_longitude"] + FEATURES].reset_index(drop=True)
print("   records kept:", len(full))


# 3. Split: train on 2016 to 2019, test on 2020
print("3. Split")
train = full[full["year"] <= 2019].reset_index(drop=True)
test = full[full["year"] == 2020].reset_index(drop=True)
common = train["seed_type"].value_counts()[lambda s: s >= 150].index          # rare seed varieties grouped, from training rows only
for part in (train, test):
    part["seed_type"] = part["seed_type"].where(part["seed_type"].isin(common), "other_variety").where(part["seed_type"].notna())
print(f"   train {len(train)} records, {train.district.nunique()} districts | test {len(test)} records")
X, y, groups = train[FEATURES], train["yield_kg_ha"].values, train["district"].values


# 4. Train: XGBoost, settings chosen by a district based search inside the training years
print("4. Train")
robust, standard = choose_scalers(train, FEATURES, CONT + SOIL)
pipe = Pipeline([("prep", build_preprocessor(robust, standard, BINARY, CAT, scale=True)),
                 ("model", xgb.XGBRegressor(tree_method="hist", n_jobs=2, random_state=SEED, n_estimators=300))])
grid_params = {"model__n_estimators": [200, 300, 400], "model__max_depth": [3, 4, 5, 6], "model__learning_rate": [0.03, 0.05, 0.1],
               "model__subsample": [0.7, 0.9], "model__colsample_bytree": [0.6, 0.8, 1.0], "model__min_child_weight": [5, 20, 50],
               "model__reg_lambda": [1, 10]}
search = RandomizedSearchCV(pipe, grid_params, n_iter=4, cv=list(GroupKFold(5).split(X, y, groups)),
                            scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1).fit(X, y)
best = search.best_estimator_
print("   settings:", search.best_params_)


# 5. Evaluate
print("5. Evaluate")
oof = cross_val_predict(best, X, y, cv=GroupKFold(5), groups=groups)           # every training district predicted by a model that never saw it
resid = y - oof
cv = {"RMSE_kg_ha": mean_squared_error(y, oof) ** 0.5, "R2": r2_score(y, oof),
      "residual_quantiles_kg_ha": {"q10": float(np.percentile(resid, 10)), "q90": float(np.percentile(resid, 90))},
      "note": "out of fold, training years 2016 to 2019, every district predicted by a model that never saw it"}
p_test = best.predict(test[FEATURES])
t = {"RMSE_kg_ha": mean_squared_error(test["yield_kg_ha"], p_test) ** 0.5, "MAE_kg_ha": mean_absolute_error(test["yield_kg_ha"], p_test),
     "R2": r2_score(test["yield_kg_ha"], p_test), "n_records": len(test)}
print(f"   cross validation: RMSE {cv['RMSE_kg_ha']:.0f}  R2 {cv['R2']:.3f}")
print(f"   2020 test:        RMSE {t['RMSE_kg_ha']:.0f}  R2 {t['R2']:.3f}")


# 6. Save what the API needs
print("6. Save")
defaults = {c: (train[c].median() if c in CONT or c in BINARY else train[c].mode().iloc[0]) for c in PRACTICE_SET}
defaults = {k: (v.item() if hasattr(v, "item") else v) for k, v in defaults.items()}
joblib.dump({"pipeline": best, "features": FEATURES, "defaults": defaults}, MODEL_DIR / "maize_model.joblib")
meta = {"version": __version__, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_records": len(train), "n_districts": int(train.district.nunique()), "years": [int(y_) for y_ in sorted(train.year.unique())],
        "target": "maize yield, kg per hectare", "features": FEATURES,
        "categories": {c: sorted(train[c].dropna().unique().tolist()) for c in CAT},
        "defaults": defaults, "best_params": search.best_params_,
        "district_blocked_estimate": cv, "test_2020": t,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "xgboost": xgb.__version__}}
(MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=1, default=float))
np.savez_compressed(MODEL_DIR / "survey_locations.npz", latlon=full[["field_latitude", "field_longitude"]].dropna().values)
print("   saved to", MODEL_DIR)


# 7. Predict: load the saved model the way the API does and make one prediction
print("7. Predict")
from maize.predict import Predictor
out = Predictor().predict(0.28, 34.75)
print("   Kakamega example:", out["predicted_yield_kg_ha"], "kg/ha, likely range", out["likely_range_kg_ha"])
