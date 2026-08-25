from __future__ import annotations

import sys

import pendulum

sys.path.append("/opt/airflow")

from airflow.decorators import dag

from utils.tasks import (
    detekcija_anomalija,
    finalne_predikcije,
    geo_i_dopunski_atributi,
    imputacija_nasumicno_nedostajucih,
    imputacija_potpuno_nedostajucih,
    priprema_vremenskih_i_diferencijalnih_atributa,
    produkcija_modela,
    ucitavanje_i_sistemske_greske,
)


@dag(
    dag_id="weatheraus_pipeline",
    description="Rain in Australia - produkcioni pipeline (Python task funkcije, izvucene iz svezaka)",
    schedule=None,  # rucno pokretanje; promeni po potrebi (npr. "@weekly")
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["weatheraus", "python-tasks"],
)
def weatheraus_pipeline():
    (
        ucitavanje_i_sistemske_greske()
        >> detekcija_anomalija()
        >> priprema_vremenskih_i_diferencijalnih_atributa()
        >> geo_i_dopunski_atributi()
        >> imputacija_potpuno_nedostajucih()
        >> imputacija_nasumicno_nedostajucih()
        >> finalne_predikcije()
        >> produkcija_modela()
    )


weatheraus_pipeline()
