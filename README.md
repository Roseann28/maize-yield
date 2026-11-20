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
3. Open `Notebook/maize_yield_analysis.ipynb` in VS Code, pick the environment as the kernel and run the cells from the top. The notebook reads `../Data/...`, so keep the folder layout as it is.

## Run the API
`docker compose up --build`, then open http://localhost:8000/docs.
Try `POST /predict` with latitude `0.28` and longitude `34.75`. Farm practices are optional. Anything left out is filled with the survey median or most common value, and the response says so. Only the listed practice names and allowed values are accepted, and anything else is rejected with a clear error.

## Results (from `Notebook/outputs`)
Training years 2016 to 2019, test year 2020, cross validation inside the training years holds out whole districts. Error is the typical size of a prediction miss, in kg per hectare (RMSE). Lower is better.

| Error (kg/ha) | Cross validation | 2020 test |
|---|---|---|
| XGBoost, practices plus soil | about 1,500 | about 1,480 |
| Always guessing the training mean | about 1,700 | about 1,560 |

* Individual field predictions are weak. The model beats the simple guess by a modest margin. Averaged to district level the predictions are much closer to the survey averages.
* Soil adds a little in cross validation (error about 23 kg/ha lower) but not on the 2020 test season.
* The model has no weather data and is trained on western Kenya. Treat predictions as estimates with a wide range, not guarantees.
* The data are observational, so the effects shown by SHAP are associations, not causes.

## Note on the saved model
`Model/maize_model.joblib` was trained earlier on all five seasons, and its stored estimate (`Model/metadata.json`) comes from that earlier setup. The notebook does not save a model, so the served model is not the same run as the notebook results above.
