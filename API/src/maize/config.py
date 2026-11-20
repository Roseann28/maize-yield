"""Paths and constants used by the API."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOIL_GRID = ROOT / "Data" / "kenya_soil_grid.npz"
MODEL_DIR = ROOT / "Model"
MODEL_FILE = MODEL_DIR / "maize_model.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"

MAX_INPUT_KG_HA = 2000        # fertiliser or lime above this is an entry error
MAX_WEEDING = 6               # weeding frequency above this is an entry error

SOIL = ["soil_ph", "soil_carbon", "soil_nitrogen", "soil_phosphorus", "soil_potassium", "soil_sand", "soil_clay"]
CONT = ["plant_doy", "plant_spacing", "row_spacing", "weed", "dap_kg_ha", "can_kg_ha", "urea_kg_ha", "npk_kg_ha",
        "lime_kg_ha", "plot_hectares", "hh_num", "hh_num_under18", "livestock"]
BINARY = ["intercrop", "compost", "pesticide", "pest_disease", "drought", "flood", "electricity"]
CAT = ["seed_type", "slope", "distance_meter", "fertility"]
PRACTICE_SET = CONT + BINARY + CAT

# plausible ranges used to validate API input (Kenya bounding box)
LAT_RANGE = (-4.9, 5.1)
LON_RANGE = (33.8, 42.0)
