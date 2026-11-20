# Soil specific maize yield prediction (Kenya)

MSc Data Science and Analytics, DSA8401 Applied Machine Learning capstone.

The project predicts maize yield (kg per hectare) from farm practices and soil properties, using the One Acre Fund Kenya survey (2016 to 2020) joined by field coordinates to a 1 km soil grid built from iSDAsoil. XGBoost is the main model, with a random forest as a comparison. A FastAPI service serves the model in Docker.

## Folders

| Folder or file | What is in it |
|---|---|
| `Data` | `kenya_soil_grid.npz` (soil layers, about 1 km). Put the survey file `one_acre_fund.xlsx` here too. It is not included. |
| `Notebook` | `maize_yield_analysis.ipynb`, the whole analysis. Its tables and figures are saved in `Notebook/outputs`. |
| `Model` | The saved model (`maize_model.joblib`), its settings (`metadata.json`) and the survey locations used for the coverage check. |
| `API` | The FastAPI service in `API/src/maize` and its tests in `API/tests`. |
| `Presentation` | Results for the report and slides: `reports`, an outline, and `docs/TECHNIQUE_JUSTIFICATION.md`. |
| `Dockerfile`, `docker-compose.yml` | Build and run the API. |
| `requirements.txt` | Libraries for the notebook, the API and the tests. |
| `AI_USE.md` | Disclosure of how AI was used. |

## Run the notebook
1. Create a virtual environment and run `pip install -r requirements.txt`.
2. Put `one_acre_fund.xlsx` in the `Data` folder.
3. Open `Notebook/maize_yield_analysis.ipynb` in VS Code, pick the environment as the kernel and run the cells from the top. The notebook reads `../Data/...`, so keep the folder layout as it is. The full run takes about 40 minutes.

## Run the API
With Docker: `docker compose up --build`, then open http://localhost:8000/docs.

Without Docker, from the project folder:
```bash
pip install -r requirements.txt
# Windows PowerShell
$env:PYTHONPATH = "API/src"
# macOS or Linux instead: export PYTHONPATH=API/src
uvicorn maize.api:app --reload
```
Try `POST /predict` with latitude `0.28` and longitude `34.75`. Farm practices are optional. Anything left out is filled with the survey median or most common value, and the response says so.

Run the tests (11) from the `API` folder with `python -m pytest`.

## Results (from `Notebook/outputs`)
Training years 2016 to 2019, test year 2020, cross validation inside the training years holds out whole districts.

| | XGBoost | Random forest |
|---|---|---|
| Cross validation R², practices plus soil | 0.215 | 0.196 |
| 2020 test R², practices plus soil | 0.101 (95% interval 0.03 to 0.17) | 0.123 |
| 2020 test R², practices only | 0.114 | 0.129 |
| 2020 test R², soil only | -0.008 | 0.023 |

* Individual field predictions are weak. Averaged to district level the cross validation R² is 0.63.
* Soil adds a little in cross validation (RMSE about 23 kg/ha lower) but not on the 2020 test season.
* The model has no weather data and is trained on western Kenya. Treat predictions as estimates with a wide range, not guarantees.
* The data are observational, so the effects shown by SHAP are associations, not causes.

## Note on the saved model
`Model/maize_model.joblib` was trained earlier on all five seasons, and its stored estimate (`Model/metadata.json`, R² 0.205, whole districts held out) comes from that earlier setup. The notebook does not save a model, so the served model is not the same run as the notebook results above.
