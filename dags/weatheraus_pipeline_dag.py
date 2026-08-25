"""
Linearan DAG koji izvrsava produkcioni podskup pipeline-a (01 -> 04 -> 05a ->
05b -> 07a -> 07b -> 08 -> 09) kao plain Python @task funkcije iz
utils/tasks.py, umesto ranijeg pristupa (nbclient izvrsavanje .ipynb fajlova
iz ../merging/notebooks) - po uzoru na
https://github.com/veljako/Airflow-Tutorial (utils/fun.py + @task + ">>").

Svaka faza i dalje cita CSV checkpoint koji je prethodna upravo napisala u
"backups/" (isti fajlovi kao ranije), pa nema grananja/paralelizma u samom
toku podataka - identicno originalnom dizajnu, samo bez nbclient posrednika.

Sam kod transformacija (utils/tasks.py, utils/common.py) je izvucen IZ
odgovarajucih svezaka u ../merging/notebooks bez izmene logike (dve
dokumentovane izuzetka - vidi napomenu na vrhu utils/tasks.py). Te sveske
ostaju NETAKNUTE u ../merging za rucno pokretanje/pregled u Jupyteru.

Namerno IZOSTAVLJENE sveske (isti razlozi kao i ranije, vidi AIRFLOW_SETUP.md
#3): 00_priprema_okruzenja (pip install je u Dockerfile.airflow, mlflow je
trajan servis), 02_eda (cista vizualizacija, ne pise nista sto naredna faza
cita), 03_train_test_split (ne pise nijedan artefakt), 06_analiza_znacaja_atributa
(dijagnostika, ne pise nista sto naredna faza cita), 10_prilog (odbaceni
eksperimenti, van produkcionog toka).

Podaci (InputData/, GeoPodaci/, backups/, logs/, reports/...) zive u ./data
ovog projekta (mount ./data:/opt/airflow/weatherdata), NE u ../merging - ovaj
projekat je samostalan, vise ne cita ../merging u radu (vidi data/README.md).
"""
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
