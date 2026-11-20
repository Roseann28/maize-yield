FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/API/src
WORKDIR /app

# xgboost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY API/requirements.txt API/requirements.txt
RUN pip install --no-cache-dir -r API/requirements.txt

COPY API/src ./API/src
COPY Model ./Model
COPY Data/kenya_soil_grid.npz ./Data/kenya_soil_grid.npz

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "maize.api:app", "--host", "0.0.0.0", "--port", "8000"]
