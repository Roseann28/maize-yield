"""FastAPI service. Run:  uvicorn maize.api:app --host 0.0.0.0 --port 8000"""
from contextlib import asynccontextmanager
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from . import __version__
from . import config as C
from .predict import Predictor

state = {}


@asynccontextmanager
async def lifespan(app):
    state["predictor"] = Predictor()
    yield
    state.clear()


app = FastAPI(title="Smallholder maize yield API (Kenya)", version=__version__, lifespan=lifespan,
              description="Predicts maize yield (kg per hectare) from a location and, optionally, farm management details. "
                          "Soil is looked up automatically from the coordinates.")


# Allowed values for the category practices. They are the categories the model saw in training
# (Model/metadata.json). If you retrain and the categories change, update these lists;
# test_category_lists_match_model in tests/test_api.py will fail to remind you.
SeedType = Literal["dk_8031", "h_614d", "h_6213", "h_6218", "local", "local_maize", "localmaize", "other_hybrid",
                   "other_variety", "otherhybrid", "pan_691", "sc_duma_43", "sc_simba_61", "wh_505", "wh_509"]
Slope = Literal["footslope", "hilltop", "midslope", "shoulder", "valley"]
Distance = Literal["1-50 m", "50-100 m", "100-200 m", "200-500 m", "500-1000 m", "1000+ m"]
Fertility = Literal["better", "same", "worse", "dont_know", "this_is_the_only_field_i_farm"]


class Practices(BaseModel):
    """The only practice names accepted are the ones listed here. An unknown name or a value outside the allowed
    list or range is rejected with a 422 error. Every field is optional; missing ones are filled with the survey
    median or most common value."""
    model_config = ConfigDict(extra="forbid")
    plant_doy: Optional[int] = Field(None, ge=1, le=181, description="Day of the year the maize was planted (Jan to Jun)")
    plant_spacing: Optional[float] = Field(None, gt=0, le=200, description="Spacing between plants, units as recorded in the survey")
    row_spacing: Optional[float] = Field(None, gt=0, le=300, description="Spacing between rows, units as recorded in the survey")
    weed: Optional[int] = Field(None, ge=0, le=C.MAX_WEEDING, description="Times weeded")
    dap_kg_ha: Optional[float] = Field(None, ge=0, le=C.MAX_INPUT_KG_HA)
    can_kg_ha: Optional[float] = Field(None, ge=0, le=C.MAX_INPUT_KG_HA)
    urea_kg_ha: Optional[float] = Field(None, ge=0, le=C.MAX_INPUT_KG_HA)
    npk_kg_ha: Optional[float] = Field(None, ge=0, le=C.MAX_INPUT_KG_HA)
    lime_kg_ha: Optional[float] = Field(None, ge=0, le=C.MAX_INPUT_KG_HA)
    plot_hectares: Optional[float] = Field(None, gt=0, le=20)
    hh_num: Optional[int] = Field(None, ge=1, le=40, description="Household size")
    hh_num_under18: Optional[int] = Field(None, ge=0, le=30)
    livestock: Optional[float] = Field(None, ge=0, le=1000, description="Cows + goats + chickens")
    intercrop: Optional[int] = Field(None, ge=0, le=1)
    compost: Optional[int] = Field(None, ge=0, le=1)
    pesticide: Optional[int] = Field(None, ge=0, le=1)
    pest_disease: Optional[int] = Field(None, ge=0, le=1)
    drought: Optional[int] = Field(None, ge=0, le=1)
    flood: Optional[int] = Field(None, ge=0, le=1)
    electricity: Optional[int] = Field(None, ge=0, le=1)
    seed_type: Optional[SeedType] = Field(None, description="Seed variety. Allowed values are listed in the schema")
    slope: Optional[Slope] = Field(None, description="Slope position of the field")
    distance_meter: Optional[Distance] = Field(None, description="Distance from the home to the field")
    fertility: Optional[Fertility] = Field(None, description="Farmer's rating of the field's fertility compared with others")


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=C.LAT_RANGE[0], le=C.LAT_RANGE[1], examples=[0.28])
    longitude: float = Field(..., ge=C.LON_RANGE[0], le=C.LON_RANGE[1], examples=[34.75])
    practices: Practices = Field(default_factory=Practices)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state, "version": __version__}


@app.get("/model-info")
def model_info():
    m = state["predictor"].meta
    return {k: m[k] for k in ("version", "trained_at", "n_records", "n_districts", "years", "target", "features", "categories",
                              "district_blocked_estimate", "versions")}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return state["predictor"].predict(req.latitude, req.longitude, req.practices.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
