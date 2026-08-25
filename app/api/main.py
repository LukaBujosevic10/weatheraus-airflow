"""
FastAPI servis koji servira predikciju kiše za sutra na osnovu svežih (uživo)
meteoroloških podataka. Nema istorijskog/2017 režima - jedini model koji se koristi
je 10-atributni model (bez Sused_RainToday_pct/Sused_Pressure3pm_avg - ta dva
zahtevaju live podatke sa SVIH susednih stanica istovremeno, a doprinose samo
~2-3% odluke modela svaki, pa je isplativije da produkcioni model bude bez njih).

Podaci se svaki put uzimaju uživo sa Open-Meteo (besplatno, bez API ključa) - videti
live_features.py za tačan mehanizam (uvek se koristi poslednji POTPUNO završen dan,
nikad tekući nezavršeni dan, jer bi dnevni agregati poput Rainfall/Sunshine bili
nepotpuni dok dan traje).

Model registruje notebooks/09_produkcija_modela.ipynb (deo Airflow DAG-a) pod imenom
WeatherAusRainModel, alias "champion". Da ga osvežiš bez restarta kontejnera posle
novog pipeline run-a, pozovi POST /reload.
"""
import os

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from live_features import ATRIBUTI_LIVE, izracunaj_live_atribute

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "WeatherAusRainModel")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
STATIONS_PATH = os.environ.get("STATIONS_PATH", "/data/GeoPodaci/stanice.csv")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

app = FastAPI(title="WeatherAus Rain Prediction API")

state = {"model": None, "threshold": 0.5, "model_version": None, "stanice": None}


def _load_model():
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    state["threshold"] = float(version.tags.get("optimalni_prag", 0.5))
    state["model_version"] = version.version


def _load_stations():
    stanice = pd.read_csv(STATIONS_PATH)
    state["stanice"] = stanice.set_index("StationName")


@app.on_event("startup")
def startup():
    _load_model()
    _load_stations()


@app.post("/reload")
def reload_all():
    """Pozovi posle novog uspešnog pipeline run-a da se ucita nova verzija modela."""
    _load_model()
    _load_stations()
    return {
        "status": "reloaded",
        "model_version": state["model_version"],
        "threshold": state["threshold"],
        "stations": len(state["stanice"]),
    }


@app.get("/health")
def health():
    stanice = state["stanice"]
    return {
        "status": "ok",
        "model_version": state["model_version"],
        "threshold": state["threshold"],
        "stations": 0 if stanice is None else len(stanice),
    }


@app.get("/stations")
def stations():
    df = state["stanice"]
    return [
        {"location": loc, "lat": float(row["Lat"]), "lon": float(row["Lon"])}
        for loc, row in df.iterrows()
    ]


@app.get("/predict/{location}")
def predict(location: str):
    stanice = state["stanice"]
    if location not in stanice.index:
        raise HTTPException(status_code=404, detail=f"Nepoznata stanica: {location}")

    lat, lon = float(stanice.loc[location, "Lat"]), float(stanice.loc[location, "Lon"])
    try:
        vrednosti, bazni_datum, nedostaju = izracunaj_live_atribute(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo upit nije uspeo: {e}")

    if nedostaju:
        raise HTTPException(
            status_code=502,
            detail=f"Open-Meteo nije vratio sve potrebne podatke za {location}: {nedostaju}",
        )

    X = pd.DataFrame([vrednosti])[ATRIBUTI_LIVE]
    verovatnoca = float(state["model"].predict_proba(X)[:, 1][0])
    pada_kisa = verovatnoca >= state["threshold"]

    return {
        "location": location,
        "date": bazni_datum,
        "features": vrednosti,
        "probability": round(verovatnoca, 4),
        "threshold": state["threshold"],
        "rain_tomorrow": "Yes" if pada_kisa else "No",
    }
