"""
Task funkcije za weatheraus_pipeline DAG - izvucene IZ notebook-ova u
../merging/notebooks (01, 04, 05a, 05b, 07a, 07b, 08, 09), bez izmene logike
transformacija/treniranja. Zamenjuju raniji pristup (nbclient izvrsavanje
.ipynb fajlova) planim Python funkcijama, po uzoru na
https://github.com/veljako/Airflow-Tutorial (utils/fun.py + @task).

Svaka funkcija odgovara tacno jednoj svesci i cita/pise iste "backups/*.csv"
checkpoint-e kao i original - DAG (dags/weatheraus_pipeline_dag.py) ih i dalje
lancano izvrsava istim redosledom (01 -> 04 -> 05a -> 05b -> 07a -> 07b -> 08 -> 09).
Sveske 02, 03, 06, 10 ostaju namerno izostavljene (isti razlozi kao i ranije -
vidi AIRFLOW_SETUP.md #3).

Poznate NAMERNE razlike u odnosu na izvorne sveske (obe dokumentovane, obe su
funkcionalne ispravke - originalne sveske u merging su OSTAVLJENE NETAKNUTE):

1. `detekcija_anomalija` (04): originalna sveska cita
   "backups/weatherAUSAfter5_2.csv" (veliko "AUS"), a fajl koji 01 stvarno pise
   je "weatherAusAfter5_2.csv" (malo "us"). Na Windows-u (case-insensitive fs)
   ovo prolazi, ali bi na Linux kontejneru (case-sensitive) pukao sa
   FileNotFoundError. Ovde je putanja ispravljena da odgovara stvarnom imenu
   fajla.
2. `imputacija_potpuno_nedostajucih` (07a) i `imputacija_nasumicno_nedostajucih`
   (07b): dijagnosticki zavrsni deo (kreirajAnalizu() - RF feature importance +
   mutual info, isti kod kao iskljucena sveska 06; kod 07b jos i R^2
   poredjenje RF-a naspram medijane + histogrami) je izostavljen - ne pise
   nista sto naredna faza cita (07b/08 citaju "backups/weatherAusAfter11_1_4.csv"
   odnosno "backups/WeatherAus_After_11_2_3_8.csv" direktno, checkpoint pre ovog
   dela), isti princip po kom je vec iskljucena sveska 06 iz DAG-a (vidi
   AIRFLOW_SETUP.md #3).
"""
from __future__ import annotations

import csv
import math
import os
import ssl
from datetime import timedelta
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from airflow.decorators import task
from sklearn.covariance import MinCovDet
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

from utils.common import (
    GEO_LIBS_AVAILABLE, SPLIT_DATE, TIMEZONEFINDER_AVAILABLE,
    calculateDiffs, calculatePreviousValues, calculateRollingMeans,
    createVaribales, gap_validan, loadData, mlflow, write_log,
)


# ---------------------------------------------------------------------------
# 01_ucitavanje_i_sistemske_greske
# ---------------------------------------------------------------------------

OGRANICENJA_VREDNOSTI = {
    "MinTemp":       (-15.0, 45.0),
    "MaxTemp":       (-10.0, 55.0),
    "Temp9am":       (-15.0, 50.0),
    "Temp3pm":       (-12.0, 55.0),
    "Rainfall":      (0.0, 600.0),
    "Evaporation":   (0.0, 30.0),
    "WindGustSpeed": (0.0, 200.0),
    "WindSpeed9am":  (0.0, 150.0),
    "WindSpeed3pm":  (0.0, 150.0),
    "Humidity9am":   (0.0, 100.0),
    "Humidity3pm":   (0.0, 100.0),
    "Pressure9am":   (950.0, 1060.0),
    "Pressure3pm":   (950.0, 1060.0),
    "Cloud9am":      (0.0, 9.0),
    "Cloud3pm":      (0.0, 9.0),
}


def _dew_point(temp_c, rh_pct):
    a, b = 17.625, 243.04
    rh = rh_pct.clip(lower=1.0, upper=100.0) / 100.0
    alpha = np.log(rh) + (a * temp_c) / (b + temp_c)
    return (b * alpha) / (a - alpha)


@task(task_id="01_ucitavanje_i_sistemske_greske", execution_timeout=timedelta(seconds=360), retries=1)
def ucitavanje_i_sistemske_greske():
    data = loadData('InputData/weatherAUS.csv')

    data = data.copy()
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfYear'] = data['Date'].dt.dayofyear
    write_log(data, "Datum tranformisan u 4 nove promenljive: godina, mesec, dan i dan u godini.", "weatherAUSAfter4.csv")

    valid_wind_dirs = {
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    }
    valid_cloud_values = set(range(0, 10))
    wind_columns = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    cloud_columns = ["Cloud9am", "Cloud3pm"]

    print("=== Detekcija anomalija u kategorijskim kolonama ===\n")
    for col in wind_columns:
        invalid = data[~data[col].isin(valid_wind_dirs)][col]
        if len(invalid) > 0:
            print(f"Kolona: {col}")
            print(f"Broj anomalnih vrednosti: {len(invalid)}")
            print(f"Jedinstvene anomalne vrednosti: {invalid.unique()}\n")
        else:
            print(f"Kolona: {col} - nema anomalija.\n")
    for col in cloud_columns:
        invalid = data[~data[col].isin(valid_cloud_values)][col]
        if len(invalid) > 0:
            print(f"Kolona: {col}")
            print(f"Broj anomalnih vrednosti: {len(invalid)}")
            print(f"Jedinstvene anomalne vrednosti: {invalid.unique()}\n")
        else:
            print(f"Kolona: {col} - nema anomalija.\n")

    write_log(data, "Detekcija kategorijskih anomalija", "weatherAusAfter5_1.csv")

    errors: dict = {}

    # R1
    mask = data["MinTemp"] > data["MaxTemp"]
    data.loc[mask, ["MinTemp", "MaxTemp"]] = np.nan
    for idx in data.index[mask]:
        errors.setdefault(idx, []).append("MinTemp > MaxTemp")

    # R2
    tol = 0.6
    for t in ("Temp9am", "Temp3pm"):
        mask = (data[t] < data["MinTemp"] - tol) | (data[t] > data["MaxTemp"] + tol)
        data.loc[mask, t] = np.nan
        for idx in data.index[mask]:
            errors.setdefault(idx, []).append(f"{t} van opsega MinTemp-MaxTemp")

    # R3 - usklađivanje RainToday sa Rainfall
    expected = np.where(data["Rainfall"] >= 1.0, "Yes", "No")
    mask = data["RainToday"] != expected
    data.loc[mask, "RainToday"] = np.nan
    for idx in data.index[mask]:
        errors.setdefault(idx, []).append("RainToday inconsistent sa Rainfall")

    # R4
    nxt_today = data.groupby("Location")["RainToday"].shift(-1)
    nxt_date = data.groupby("Location")["Date"].shift(-1)
    contiguous = (nxt_date - data["Date"]).dt.days.eq(1)
    mask = contiguous & (data["RainTomorrow"] != nxt_today)
    data.loc[mask, "RainTomorrow"] = np.nan
    for idx in data.index[mask]:
        errors.setdefault(idx, []).append("RainTomorrow inconsistent sa RainToday sledeceg dana")

    # R5 - duplikati
    mask = data.duplicated(subset=["Location", "Date"], keep="first")
    data.loc[mask, :] = np.nan
    for idx in data.index[mask]:
        errors.setdefault(idx, []).append("Dupli zapis")

    # R6 - opsezi
    for v, (low, high) in OGRANICENJA_VREDNOSTI.items():
        mask = (data[v] < low) | (data[v] > high)
        data.loc[mask, v] = np.nan
        for idx in data.index[mask]:
            errors.setdefault(idx, []).append(f"{v} van opsega [{low}, {high}]")

    # R7
    mx = data[["WindSpeed9am", "WindSpeed3pm"]].max(axis=1)
    mask = (data["WindGustSpeed"] < mx).fillna(False)
    data.loc[mask, "WindGustSpeed"] = np.nan
    for idx in data.index[mask]:
        errors.setdefault(idx, []).append("WindGustSpeed < max(WindSpeed9am, WindSpeed3pm)")

    # R8
    for t, h in (("Temp9am", "Humidity9am"), ("Temp3pm", "Humidity3pm")):
        td = _dew_point(data[t], data[h])
        mask = (td > data[t]).fillna(False)
        data.loc[mask, t] = np.nan
        for idx in data.index[mask]:
            errors.setdefault(idx, []).append(f"Dew point > {t}")

    # R9
    mask = ((data["Pressure3pm"] - data["Pressure9am"]).abs() > 12.0).fillna(False)
    data.loc[mask, ["Pressure9am", "Pressure3pm"]] = np.nan
    for idx in data.index[mask]:
        errors.setdefault(idx, []).append("Nedozvoljeni skok pritiska")

    # R10
    gap = data.groupby("Location")["Date"].diff().dt.days
    mask = gap > 1
    for idx in data.index[mask]:
        errors.setdefault(idx, []).append("Nedostaje dan u seriji")

    write_log(data, "Uklonjene fizicke anomalije", "weatherAusAfter5_2.csv")


# ---------------------------------------------------------------------------
# 04_detekcija_anomalija
# ---------------------------------------------------------------------------

@task(task_id="04_detekcija_anomalija", execution_timeout=timedelta(seconds=960), retries=1)
def detekcija_anomalija():
    # NAPOMENA: original cita "weatherAUSAfter5_2.csv" (veliko AUS) - ispravljeno,
    # vidi objasnjenje na vrhu fajla (tacka 1).
    data = loadData("backups/weatherAusAfter5_2.csv")
    is_train = data['Date'] < SPLIT_DATE
    identifiers = data[['Location', 'Date']].copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['Month', 'Year', "Day"]]

    final_results = identifiers.copy()

    for target_col in numeric_cols:
        data2 = data[['Location', 'Month', target_col]]
        train_data = data2[is_train].dropna()

        if train_data.empty:
            continue

        grouped = train_data.groupby(['Location', 'Month'])[target_col]

        median_lookup = grouped.median()
        mad_lookup = grouped.apply(lambda x: np.median(np.abs(x - np.median(x))))
        q25_lookup = grouped.quantile(0.25)
        q75_lookup = grouped.quantile(0.75)

        global_median = train_data[target_col].median()
        global_mad = np.median(np.abs(train_data[target_col] - global_median))
        global_q25 = train_data[target_col].quantile(0.25)
        global_q75 = train_data[target_col].quantile(0.75)

        key = list(zip(data2['Location'], data2['Month']))
        median = pd.Series([median_lookup.get(k, global_median) for k in key], index=data2.index)
        mad = pd.Series([mad_lookup.get(k, global_mad) for k in key], index=data2.index)
        q25 = pd.Series([q25_lookup.get(k, global_q25) for k in key], index=data2.index)
        q75 = pd.Series([q75_lookup.get(k, global_q75) for k in key], index=data2.index)

        mad_safe = mad.replace(0, 1e-6)
        iqr_safe = (q75 - q25).replace(0, 1e-6)

        mod_z_score = 0.6745 * (data2[target_col] - median) / mad_safe
        iqr_score = (data2[target_col] - median) / iqr_safe

        valid = data2[target_col].notna()
        final_results.loc[valid, f'{target_col}_IQR_Score'] = iqr_score[valid]
        final_results.loc[valid, f'{target_col}_ModZ_Score'] = mod_z_score[valid]

    final_results.to_csv('AnomalyDetectionResults/skorovi_univarijatni.csv', index=False)
    print("Svi lokalizovani univarijatni skorovi su sačuvani u 'AnomalyDetectionResults/skorovi_univarijatni.csv'.")

    # --- 8.2 Multivarijantna analiza (Mahalanobis, KNN, Isolation Forest) ---
    existing_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindSpeed9am']
    results = data[['Location', 'Date']].copy()
    data_imputed_df = data.copy()

    for col in existing_cols:
        loc_month_median = data_imputed_df.loc[is_train].groupby(['Location', 'Month'])[col].median()
        loc_median = data_imputed_df.loc[is_train].groupby('Location')[col].median()
        global_median = data_imputed_df.loc[is_train, col].median()

        key = list(zip(data_imputed_df['Location'], data_imputed_df['Month']))
        fill_lm = pd.Series([loc_month_median.get(k, np.nan) for k in key], index=data_imputed_df.index)
        fill_l = data_imputed_df['Location'].map(loc_median)
        data_imputed_df[col] = data_imputed_df[col].fillna(fill_lm).fillna(fill_l).fillna(global_median)

    scaler = StandardScaler()
    train_mat = scaler.fit_transform(data_imputed_df.loc[is_train, existing_cols])
    data_mat = scaler.transform(data_imputed_df[existing_cols])

    robust_cov = MinCovDet(support_fraction=0.75, random_state=42).fit(train_mat)
    mahalanobis_distances = np.sqrt(robust_cov.mahalanobis(data_mat))
    results['Mahalanobis_Score'] = mahalanobis_distances

    k = 5
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(train_mat)
    distances, _ = nbrs.kneighbors(data_mat)
    self_match = np.isclose(distances[:, 0], 0.0)
    knn_dist_bez_sebe = np.where(self_match[:, None], distances[:, 1:], distances[:, :-1])
    results['KNN_Score'] = knn_dist_bez_sebe.mean(axis=1)

    iso_forest = IsolationForest(random_state=42)
    iso_forest.fit(train_mat)
    results['IForest_Score'] = -iso_forest.decision_function(data_mat)

    results.to_csv('AnomalyDetectionResults/skorovi_multi.csv', index=False)
    print("Multivarijatni skorovi su sačuvani u 'AnomalyDetectionResults/skorovi_multi.csv'.")

    # --- LOF (uzorak od 40.000 elemenata iz train dela) ---
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = data[['Location', 'Date'] + numeric_cols].copy()

    df_valid = df_numeric.dropna().copy()
    is_train_valid = is_train.reindex(df_valid.index)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_valid.loc[is_train_valid, numeric_cols])
    scaled_data = scaler.transform(df_valid[numeric_cols])

    print("Treniranje LOF modela na uzorku (samo iz train dela)...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(train_scaled), min(40000, len(train_scaled)), replace=False)
    sample_data = train_scaled[sample_indices]

    lof = LocalOutlierFactor(n_neighbors=35, novelty=True)
    lof.fit(sample_data)
    lof_scores = -lof.score_samples(scaled_data)

    lof_results = df_valid[['Location', 'Date']].copy()
    lof_results['LOF_Score'] = lof_scores
    lof_results.to_csv('AnomalyDetectionResults/skorovi_lof.csv', index=False)
    print("Kontinualni LOF skorovi su izvezeni u 'AnomalyDetectionResults/skorovi_lof.csv'.")

    # --- 8.3 Finalna detekcija i brisanje anomalije ---
    df_uni = pd.read_csv('AnomalyDetectionResults/skorovi_univarijatni.csv')
    df_lof = pd.read_csv('AnomalyDetectionResults/skorovi_lof.csv')
    df_multi = pd.read_csv('AnomalyDetectionResults/skorovi_multi.csv')

    for d in [df_uni, df_lof, df_multi]:
        d['Date'] = pd.to_datetime(d['Date'], errors='coerce')

    merged = df_uni.merge(df_multi, on=['Location', 'Date'], how='left')
    merged = merged.merge(df_lof, on=['Location', 'Date'], how='left')

    score_cols = [c for c in merged.columns if c not in ['Location', 'Date']]

    def dobij_tezinu(ime_kolone):
        if 'ModZ' in ime_kolone or 'Mahalanobis' in ime_kolone or 'IQR' in ime_kolone:
            return 1.5
        if 'Z_Score' in ime_kolone:
            return 1.2
        if 'IForest' in ime_kolone or 'LOF' in ime_kolone or 'KNN' in ime_kolone:
            return 1.0
        return 1.0

    final_scores = pd.Series(0.0, index=merged.index)
    total_weights = pd.Series(0.0, index=merged.index)

    for col in score_cols:
        w = dobij_tezinu(col)
        rank = merged[col].rank(pct=True)
        boosted = -np.log1p(-rank.clip(upper=0.9999))
        valid_mask = merged[col].notna()
        final_scores[valid_mask] += boosted[valid_mask] * w
        total_weights[valid_mask] += w

    merged['Anomalijski_Skor'] = final_scores / total_weights.replace(0, np.nan)

    merged_is_train = merged['Date'] < SPLIT_DATE
    prag = merged.loc[merged_is_train, 'Anomalijski_Skor'].quantile(0.999)
    merged['JE_ANOMALIJA'] = merged['Anomalijski_Skor'] > prag
    obrisi = merged['JE_ANOMALIJA'] & merged_is_train

    final_report = merged[merged['JE_ANOMALIJA']].sort_values(by='Anomalijski_Skor', ascending=False)
    final_report.to_csv('AnomalyDetectionResults/konacni_izveštaj_anomalija.csv', index=False)
    broj_anomalija = len(final_report)
    broj_obrisanih = int(obrisi.sum())
    print(f"\nGotovo! Prag detekcije je postavljen na: {prag:.4f} (racunat samo na train delu)")
    print(f"Pronađeno je ukupno {broj_anomalija} anomalija (u celom skupu), od cega ce biti obrisano {broj_obrisanih} (samo iz train dela).")
    print("Detaljan spisak pravih anomalija izvezen je u 'AnomalyDetectionResults/konacni_izveštaj_anomalija.csv'.")

    mergedInfo = merged[['Location', 'Date']].copy()
    mergedInfo['OBRISI'] = obrisi.values
    cd_join = data.merge(mergedInfo, on=['Location', 'Date'], how='left')

    cleaned_without_anoms = cd_join[cd_join['OBRISI'] != True].copy()
    cleaned_without_anoms.drop(columns=['OBRISI'], inplace=True)

    write_log(cleaned_without_anoms, "Obrisane anomalije (samo iz train dela)", "weatherAusAfter8_3.csv")


# ---------------------------------------------------------------------------
# 05a_priprema_vremenskih_i_diferencijalnih_atributa
# ---------------------------------------------------------------------------

_WIND_DIR_MAP = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
}


def _wind_dir_to_sin_cos(df, col_name):
    df[col_name + '_deg'] = df[col_name].map(_WIND_DIR_MAP)
    df[col_name + '_sin'] = np.sin(np.radians(df[col_name + '_deg'])).round(2)
    df[col_name + '_cos'] = np.cos(np.radians(df[col_name + '_deg'])).round(2)
    df.drop(columns=[col_name, col_name + '_deg'], inplace=True)
    return df


def _convertWinds(df):
    df = _wind_dir_to_sin_cos(df, 'WindDir3pm')
    df = _wind_dir_to_sin_cos(df, 'WindDir9am')
    df = _wind_dir_to_sin_cos(df, 'WindGustDir')
    return df


@task(task_id="05a_priprema_vremenskih_i_diferencijalnih_atributa", execution_timeout=timedelta(seconds=360), retries=1)
def priprema_vremenskih_i_diferencijalnih_atributa():
    data = loadData('backups/weatherAusAfter8_3.csv')
    is_train = data['Date'] < SPLIT_DATE

    data['Year_Scaled'] = (data['Year'] - data.loc[is_train, 'Year'].min()) / (data.loc[is_train, 'Year'].max() - data.loc[is_train, 'Year'].min())
    data['DayOfYear_Sin'] = np.sin(2 * np.pi * data['Date'].dt.dayofyear / 365.25).round(2)
    data['DayOfYear_Cos'] = np.cos(2 * np.pi * data['Date'].dt.dayofyear / 365.25).round(2)
    data.drop(columns=['Year', "Month", "Day", "DayOfYear"], inplace=True)
    write_log(data, "Transformacija vremenskih podataka", "weatherAusAfter9_1_1.csv")

    data = _convertWinds(data)
    write_log(data, "Transformacija podataka o vetru", "weatherAusAfter9_1_2.csv")

    data = calculateDiffs(data)
    write_log(data, "Dodavanje diferencijalnih atributa", "weatherAusAfter9_2_1.csv")

    data = calculateRollingMeans(data)
    write_log(data, "Dodavanje pokretnih proseka", "weatherAusAfter9_2_2.csv")

    data = calculatePreviousValues(data)
    write_log(data, "Dodavanje lag podataka", "weatherAusAfter9_2_3.csv")


# ---------------------------------------------------------------------------
# 05b_geo_i_dopunski_atributi
# ---------------------------------------------------------------------------

_BOM_ID_MAP = {
    'Adelaide': '023090', 'Sydney': '066062', 'Melbourne': '086071',
    'Brisbane': '040913', 'Perth': '009021', 'Hobart': '094029',
    'Darwin': '014015', 'Canberra': '070351', 'SydneyAirport': '066037',
    'MelbourneAirport': '086282', 'PerthAirport': '009014', 'AliceSprings': '015590',
    'Newcastle': '061055', 'Wollongong': '068188', 'Uluru': '015643',
    'GoldCoast': '040764', 'Townsville': '032040', 'Cairns': '031011',
}

_IME_IZUZECI = {
    'Nhil': 'NHILL', 'PearceRAAF': 'PEARCE RAAF', 'BadgerysCreek': 'BADGERYS CREEK',
    'CoffsHarbour': 'COFFS HARBOUR', 'MountGinini': 'MOUNT GININI',
    'MountGambier': 'MOUNT GAMBIER', 'NorfolkIsland': 'NORFOLK ISLAND',
    'WaggaWagga': 'WAGGA WAGGA',
}

_CLIMATE_MAPPING = {
    'Darwin': 'Tropical', 'Cairns': 'Tropical', 'Katherine': 'Tropical',
    'Brisbane': 'Subtropical', 'GoldCoast': 'Subtropical', 'CoffsHarbour': 'Subtropical',
    'Newcastle': 'Subtropical', 'NorahHead': 'Subtropical', 'Sydney': 'Subtropical',
    'SydneyAirport': 'Subtropical', 'Wollongong': 'Subtropical', 'Williamtown': 'Subtropical',
    'BadgerysCreek': 'Subtropical', 'Penrith': 'Subtropical', 'Richmond': 'Subtropical',
    'Perth': 'Subtropical', 'PerthAirport': 'Subtropical', 'PearceRAAF': 'Subtropical',
    'NorfolkIsland': 'Subtropical',
    'AliceSprings': 'Desert', 'Woomera': 'Desert',
    'Cobar': 'Grassland', 'Moree': 'Grassland', 'Mildura': 'Grassland', 'SalmonGums': 'Grassland',
    'Melbourne': 'Temperate', 'MelbourneAirport': 'Temperate', 'Ballarat': 'Temperate',
    'Bendigo': 'Temperate', 'Watsonia': 'Temperate', 'Adelaide': 'Temperate',
    'MountGambier': 'Temperate', 'Nuriootpa': 'Temperate', 'Canberra': 'Temperate',
    'Tuggeranong': 'Temperate', 'MountGinini': 'Temperate', 'Hobart': 'Temperate',
    'Launceston': 'Temperate', 'Albury': 'Temperate', 'WaggaWagga': 'Temperate',
    'Albany': 'Temperate', 'Witchcliffe': 'Temperate', 'Walpole': 'Temperate',
    'Dartmoor': 'Temperate', 'Portland': 'Temperate', 'Sale': 'Temperate',
    "Uluru": "Desert", "Townsville": "Tropical", "Nhil": "Temperate",
}

_MAPIRANJE_LOKACIJA_ROSE = {
    'YABA': 'Albany', 'YAYE': 'Uluru', 'YBAS': 'AliceSprings',
    'YBBN': 'Brisbane', 'YBCS': 'Cairns', 'YBTL': 'Townsville',
    'YCBA': 'Cobar', 'YCFS': 'CoffsHarbour', 'YMAY': 'Albury',
    'YMEN': 'Melbourne', 'YMHB': 'Hobart', 'YMIA': 'Mildura',
    'YMOR': 'Moree', 'YMTG': 'MountGambier', 'YPAD': 'Adelaide',
    'YPDN': 'Darwin', 'YPJT': 'Perth', 'YPTN': 'Katherine',
    'YPWR': 'Woomera', 'YSCB': 'Canberra', 'YSNF': 'NorfolkIsland',
    'YSWG': 'WaggaWagga', 'YWLM': 'Williamtown',
}


def _prilagodi_ime(ime):
    import re
    ime_string = str(ime).strip()
    if ime_string in _IME_IZUZECI:
        return _IME_IZUZECI[ime_string]
    ime_odvojeno = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', ime_string)).strip().upper()
    ime_odvojeno = re.sub(r'\s+', ' ', ime_odvojeno)
    return ime_odvojeno


def _ucitaj_geografske_karakteristike(df_stanice):
    colspecs = [(0, 7), (7, 13), (13, 54), (54, 62), (62, 70), (70, 79), (79, 89), (89, 104), (104, 108), (108, 119), (119, 128), (128, 134)]
    imena_kolona = ["Site", "Dist", "Site_name", "Start", "End", "Lat", "Lon", "Source", "STA", "Height", "Bar_ht", "WMO"]

    df_stations = pd.read_fwf('InputData/stations.txt', skiprows=4, colspecs=colspecs, names=imena_kolona)
    df_stations['Site_name'] = df_stations['Site_name'].str.strip().str.upper()
    df_stations['Site'] = df_stations['Site'].astype(str).str.zfill(6)
    df_stations = df_stations[df_stations['Height'] != '..']

    lat_lista, lon_lista, height_lista = [], [], []
    ime_kolone = 'StationName'

    for _, row in df_stanice.iterrows():
        kaggle_ime = str(row[ime_kolone]).strip()

        if kaggle_ime in _BOM_ID_MAP:
            trazeni_id = _BOM_ID_MAP[kaggle_ime]
            poklapanja = df_stations[df_stations['Site'] == trazeni_id]
        else:
            trazeno_ime = _prilagodi_ime(kaggle_ime)
            poklapanja = df_stations[
                (df_stations['Site_name'] == trazeno_ime) |
                (df_stations['Site_name'].str.startswith(trazeno_ime + ' ', na=False))
            ]
            if not poklapanja.empty:
                aero_stanice = poklapanja[poklapanja['Site_name'].str.contains('AIRPORT|AERO|AWS|AMO|OBSERVATORY|MO', na=False)]
                if not aero_stanice.empty:
                    poklapanja = aero_stanice

        if not poklapanja.empty:
            najbolja_stanica = poklapanja.iloc[0]
            lat_lista.append(najbolja_stanica['Lat'])
            lon_lista.append(najbolja_stanica['Lon'])
            height_lista.append(najbolja_stanica['Height'])
        else:
            lat_lista.append(None)
            lon_lista.append(None)
            height_lista.append(None)
            print(f"Upozorenje: Nije pronađena adekvatna stanica za '{kaggle_ime}'")

    df_stanice['Lat'] = lat_lista
    df_stanice['Lon'] = lon_lista
    df_stanice['Height'] = height_lista
    df_stanice['Lat'] = df_stanice['Lat'].astype(float)
    df_stanice['Lon'] = df_stanice['Lon'].astype(float)
    df_stanice['Height'] = df_stanice['Height'].astype(float)

    df_stanice[["StationName", "Lat", "Lon", "Height"]].to_csv("GeoPodaci/stanice.csv", index=False)
    return df_stanice


def _izracunaj_udaljenost_od_okeana(df_stanice):
    if GEO_LIBS_AVAILABLE:
        import geopandas as gpd
        from shapely.geometry import Point, LineString
        import cartopy.io.shapereader as shpreader
        import pyproj
        from shapely.ops import transform, linemerge

        ssl._create_default_https_context = ssl._create_unverified_context

        geometrija = [Point(xy) for xy in zip(df_stanice['Lon'], df_stanice['Lat'])]
        gdf_stanice = gpd.GeoDataFrame(df_stanice, geometry=geometrija)
        gdf_stanice.set_crs(epsg=4326, inplace=True)

        shpfilename = shpreader.natural_earth(resolution='10m', category='physical', name='coastline')
        obale_kolekcija = list(shpreader.Reader(shpfilename).geometries())
        australija_obale = [geom for geom in obale_kolekcija if geom.bounds[0] > 110 and geom.bounds[2] < 160 and geom.bounds[1] > -50 and geom.bounds[3] < 0]

        if len(australija_obale) > 1:
            kombinovana_obala = linemerge(australija_obale)
        else:
            kombinovana_obala = australija_obale[0]

        gdf_stanice = gdf_stanice.to_crs(epsg=3577)
        projekcija_wgs84 = pyproj.CRS('EPSG:4326')
        projekcija_aus = pyproj.CRS('EPSG:3577')
        projektor = pyproj.Transformer.from_crs(projekcija_wgs84, projekcija_aus, always_xy=True).transform
        obala_metricka = transform(projektor, kombinovana_obala)

        def izracunaj_usmerenu_udaljenost(tacka, pravac):
            x, y = tacka.x, tacka.y
            duzina_zraka = 5000000
            if pravac == 'Sever':
                krajnja_tacka = (x, y + duzina_zraka)
            elif pravac == 'Jug':
                krajnja_tacka = (x, y - duzina_zraka)
            elif pravac == 'Istok':
                krajnja_tacka = (x + duzina_zraka, y)
            elif pravac == 'Zapad':
                krajnja_tacka = (x - duzina_zraka, y)
            zrak = LineString([(x, y), krajnja_tacka])
            presek = zrak.intersection(obala_metricka)
            if presek.is_empty:
                return 5000.0
            return tacka.distance(presek) / 1000

        df_stanice['Dist_Sever_km'] = gdf_stanice['geometry'].apply(lambda pt: izracunaj_usmerenu_udaljenost(pt, 'Sever'))
        df_stanice['Dist_Jug_km'] = gdf_stanice['geometry'].apply(lambda pt: izracunaj_usmerenu_udaljenost(pt, 'Jug'))
        df_stanice['Dist_Istok_km'] = gdf_stanice['geometry'].apply(lambda pt: izracunaj_usmerenu_udaljenost(pt, 'Istok'))
        df_stanice['Dist_Zapad_km'] = gdf_stanice['geometry'].apply(lambda pt: izracunaj_usmerenu_udaljenost(pt, 'Zapad'))

        df_stanice['Inv_Dist_Sever'] = 1 / (df_stanice['Dist_Sever_km'] + 1)
        df_stanice['Inv_Dist_Jug'] = 1 / (df_stanice['Dist_Jug_km'] + 1)
        df_stanice['Inv_Dist_Istok'] = 1 / (df_stanice['Dist_Istok_km'] + 1)
        df_stanice['Inv_Dist_Zapad'] = 1 / (df_stanice['Dist_Zapad_km'] + 1)

        return pd.DataFrame(df_stanice.drop(columns='geometry', errors='ignore'))

    print("[info] Preskacem racunanje udaljenosti od okeana (geopandas/cartopy nedostupni),"
          " ucitavam vec sacuvane vrednosti...")
    _cache = pd.read_csv("backups/weatherAusAfter9_2_5_4.csv")
    _lookup_cols = ['Location', 'Inv_Dist_Sever', 'Inv_Dist_Jug', 'Inv_Dist_Istok', 'Inv_Dist_Zapad']
    _lookup = _cache[_lookup_cols].drop_duplicates(subset=['Location']).rename(columns={'Location': 'StationName'})
    return df_stanice.merge(_lookup, on='StationName', how='left')


def _pripremi_spojen_df(df_main, df_elev):
    df = df_main.copy()
    elev_dict = df_elev.set_index('StationName')['Height'].to_dict()
    lat_dict = df_elev.set_index("StationName")["Lat"].to_dict()
    lon_dict = df_elev.set_index("StationName")["Lon"].to_dict()
    distanca_okean = {}
    for x in ["Inv_Dist_Sever", "Inv_Dist_Jug", "Inv_Dist_Istok", "Inv_Dist_Zapad"]:
        df_elev[x] = df_elev[x].round(4)
        distanca_okean[x] = df_elev.set_index('StationName')[x].to_dict()
    klima_dict = df_elev.set_index('StationName')['ClimateGroup'].to_dict()

    df['Nadmorska visina (m)'] = df['Location'].map(elev_dict)
    df['Nadmorska visina (m)'] = df['Nadmorska visina (m)'].astype(float)
    for x in ["Inv_Dist_Sever", "Inv_Dist_Jug", "Inv_Dist_Istok", "Inv_Dist_Zapad"]:
        df[x] = df['Location'].map(distanca_okean[x])
    df['klima'] = df['Location'].map(klima_dict)
    df["Lat"] = df["Location"].map(lat_dict)
    df["Lon"] = df["Location"].map(lon_dict)
    df['Lat'] = df['Lat'].astype(float)
    df['Lon'] = df['Lon'].astype(float)
    df = pd.get_dummies(df, columns=['klima'], drop_first=True, dtype=int)
    return df


def _obradi_tacku_rose(ulazni_fajl, izlazni_fajl):
    import pytz
    from timezonefinder import TimezoneFinder

    df = pd.read_csv(ulazni_fajl)
    df = df.rename(columns={'station': 'Lokacija', 'valid': 'Vreme_UTC', 'lat': 'Lat', 'lon': 'Lon', 'dwpc': 'TackaRose'})
    df['Vreme_UTC'] = pd.to_datetime(df['Vreme_UTC'], utc=True)

    tf = TimezoneFinder()
    jedinstvene_lok = df[['Lokacija', 'Lat', 'Lon']].drop_duplicates()

    def nadji_zonu(lat, lon):
        zona = tf.timezone_at(lng=lon, lat=lat)
        return zona if zona else 'UTC'

    jedinstvene_lok['Zona'] = jedinstvene_lok.apply(lambda row: nadji_zonu(row['Lat'], row['Lon']), axis=1)
    df = df.merge(jedinstvene_lok[['Lokacija', 'Zona']], on='Lokacija', how='left')

    df['Vreme_Lokalno'] = pd.NaT
    for zona, grupa in df.groupby('Zona'):
        lokalna_zona = pytz.timezone(zona)
        df.loc[grupa.index, 'Vreme_Lokalno'] = grupa['Vreme_UTC'].dt.tz_convert(lokalna_zona).dt.tz_localize(None)

    df['Datum'] = df['Vreme_Lokalno'].dt.date
    df['Idealno_9am'] = pd.to_datetime(df['Datum'].astype(str) + ' 09:00:00')
    df['Idealno_3pm'] = pd.to_datetime(df['Datum'].astype(str) + ' 15:00:00')
    df['Razlika_9am'] = (df['Vreme_Lokalno'] - df['Idealno_9am']).abs()
    df['Razlika_3pm'] = (df['Vreme_Lokalno'] - df['Idealno_3pm']).abs()

    max_tolerancija = pd.Timedelta(hours=2)

    df_9am = df[df['Razlika_9am'] <= max_tolerancija].copy()
    df_9am = df_9am.sort_values('Razlika_9am').drop_duplicates(subset=['Lokacija', 'Datum'])
    df_9am = df_9am.rename(columns={'TackaRose': 'TackaRose9am', 'Vreme_UTC': 'univerzalnoVremeZa9am'})

    df_3pm = df[df['Razlika_3pm'] <= max_tolerancija].copy()
    df_3pm = df_3pm.sort_values('Razlika_3pm').drop_duplicates(subset=['Lokacija', 'Datum'])
    df_3pm = df_3pm.rename(columns={'TackaRose': 'TackaRose3pm', 'Vreme_UTC': 'univerzalnoVremeZa3pm'})

    izlaz_df = pd.merge(
        df_9am[['Lokacija', 'Datum', 'TackaRose9am', 'univerzalnoVremeZa9am']],
        df_3pm[['Lokacija', 'Datum', 'TackaRose3pm', 'univerzalnoVremeZa3pm']],
        on=['Lokacija', 'Datum'], how='outer',
    )
    izlaz_df = izlaz_df.sort_values(['Lokacija', 'Datum'])
    izlaz_df.to_csv(izlazni_fajl, index=False)
    print(f"Gotovo! Podaci su sačuvani u: {izlazni_fajl}")


@task(task_id="05b_geo_i_dopunski_atributi", execution_timeout=timedelta(seconds=960), retries=1)
def geo_i_dopunski_atributi():
    data = loadData("backups/weatherAusAfter9_2_3.csv")

    unique_stations = data['Location'].unique()
    df_stanice = pd.DataFrame(unique_stations, columns=['StationName'])
    df_stanice = _ucitaj_geografske_karakteristike(df_stanice)

    df_rezultat = _izracunaj_udaljenost_od_okeana(df_stanice)
    df_rezultat['ClimateGroup'] = df_rezultat['StationName'].map(_CLIMATE_MAPPING)

    data = _pripremi_spojen_df(data, df_rezultat)
    write_log(data, "Ucitavanje podataka o stanicana", "weatherAusAfter9_2_5_4.csv")

    if TIMEZONEFINDER_AVAILABLE and not os.path.exists('reports/tacka_rose.csv'):
        _obradi_tacku_rose('InputData/asos.csv', 'reports/tacka_rose.csv')
    else:
        print("[info] reports/tacka_rose.csv vec postoji (ili timezonefinder nedostupan) -"
              " preskacem ponovno racunanje tacke rose.")

    df_tacka_rose = pd.read_csv("reports/tacka_rose.csv")
    df_tacka_rose['StationName'] = df_tacka_rose['Lokacija'].map(_MAPIRANJE_LOKACIJA_ROSE)

    data['Date'] = pd.to_datetime(data['Date'])
    df_tacka_rose['Datum'] = pd.to_datetime(df_tacka_rose['Datum'])
    data = pd.merge(
        data, df_tacka_rose,
        left_on=['Location', 'Date'], right_on=['StationName', 'Datum'], how='left',
    )
    data = data.drop(columns=['StationName', 'Datum', 'Lokacija', "univerzalnoVremeZa9am", "univerzalnoVremeZa3pm"])
    write_log(data, "Dodavanje podataka o tacki rose", "weatherAusAfter9_2_5.csv")

    indikatori = data.isna().astype(int).add_suffix('_missing')
    indikatori = indikatori.loc[:, indikatori.any()]
    df_prosiren = pd.concat([data, indikatori], axis=1)
    write_log(df_prosiren, "Dodavanje indikatora o nedostajucim podacima", "weatherAusAfter9_2_6.csv")


# ---------------------------------------------------------------------------
# 07a_imputacija_potpuno_nedostajucih
# ---------------------------------------------------------------------------

def _haversine_razdaljina(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _generisi_razdaljine(ulazni_fajl, izlazni_fajl):
    stanice = []
    with open(ulazni_fajl, mode='r', encoding='utf-8') as fajl:
        citac = csv.DictReader(fajl)
        for red in citac:
            stanice.append({'ime': red['StationName'], 'lat': float(red['Lat']), 'lon': float(red['Lon'])})
    print(f"Uspešno učitano {len(stanice)} stanica iz fajla '{ulazni_fajl}'.")
    brojac_parova = 0
    with open(izlazni_fajl, mode='w', encoding='utf-8', newline='') as izlaz:
        polja = ['stanica_1', 'stanica_2', 'razdaljina_km']
        pisac = csv.DictWriter(izlaz, fieldnames=polja)
        pisac.writeheader()
        for stanica1, stanica2 in combinations(stanice, 2):
            razdaljina = _haversine_razdaljina(stanica1['lat'], stanica1['lon'], stanica2['lat'], stanica2['lon'])
            pisac.writerow({'stanica_1': stanica1['ime'], 'stanica_2': stanica2['ime'], 'razdaljina_km': round(razdaljina, 2)})
            brojac_parova += 1
    print(f"Završeno! Izračunato i upisano {brojac_parova} razdaljina u fajl '{izlazni_fajl}'.")


def _pronadji_grupe_lokacija(ime_fajla, pragovi):
    df = pd.read_csv(ime_fajla)
    rezultati = {}
    for i in range(1, len(pragovi)):
        pragMin = pragovi[i - 1]
        prag = pragovi[i]
        G = nx.Graph()
        for _, red in df.iterrows():
            lok1, lok2, udaljenost = red['stanica_1'], red['stanica_2'], red['razdaljina_km']
            if udaljenost <= prag and udaljenost > pragMin:
                G.add_edge(lok1, lok2)
        sve_grupe = list(nx.find_cliques(G))
        filtrirane_grupe = [grupa for grupa in sve_grupe if len(grupa) > 1]
        filtrirane_grupe.sort(key=len, reverse=True)
        rezultati[prag] = filtrirane_grupe
    return rezultati


def _generisi_matricu_korelacija_po_grupama_i_atributima(data, razdaljina, nedostajuci_atributi, pronadjene_grupe):
    data = data[data['Date'] < SPLIT_DATE]
    numericki_atributi = [x for x in set(nedostajuci_atributi) if x in data.select_dtypes(include=[np.number]).columns.tolist()]
    grupe = pronadjene_grupe.get(razdaljina, [])
    if not grupe:
        print(f"  Nema pronađenih grupa za razdaljinu {razdaljina} km. Preskačem...")
        return {}

    rezultati = {}
    for grupa in grupe:
        rezultati[";".join(grupa)] = {}
        df_grupa = data[data['Location'].isin(grupa)]
        for atribut in numericki_atributi:
            pivot = df_grupa.pivot(index='Date', columns='Location', values=atribut)
            corr_matrix = pivot.corr().abs()
            np.fill_diagonal(corr_matrix.values, np.nan)
            mean_corr = corr_matrix.stack().mean(skipna=True)
            if pd.notna(mean_corr) and mean_corr >= 0.0:
                rezultati[";".join(grupa)][atribut] = mean_corr
            else:
                rezultati[";".join(grupa)][atribut] = np.nan

    naziv_fajla = f'reports/claster_corelations/korelacije_po_grupama_i_atributima_razdljina_{razdaljina}.csv'
    df_matrica = pd.DataFrame([{"Grupa": k, **v} for k, v in rezultati.items()]).set_index("Grupa")
    df_matrica.to_csv(naziv_fajla)
    print(f"Matrica uspešno sačuvana kao '{naziv_fajla}'.\n")
    return rezultati


def _popuni_po_grupama(df, grupe, atribut, min_corr, korelacije):
    df_copy = df.copy()
    uspesanXGBoost = 0
    neuspesanXGBoost = 0
    kljucevi = korelacije.keys() if korelacije else []

    udaljenosti_dict = {}
    if atribut not in ['Cloud3pm', 'Cloud9am']:
        df_dist = pd.read_csv('GeoPodaci/udaljenost_stanica.csv')
        for _, row in df_dist.iterrows():
            par = tuple(sorted([row['stanica_1'], row['stanica_2']]))
            udaljenosti_dict[par] = row['razdaljina_km']

    for grupa in grupe:
        df_local = df_copy[df_copy['Location'].isin(grupa)]
        if df_local.empty:
            continue

        for loc in grupa:
            maska_lokacija = (df_copy['Location'] == loc)
            ukupno_merenja = maska_lokacija.sum()
            maska_nedostajuci = maska_lokacija & (df_copy[atribut].isna())
            nedostaje_merenja = maska_nedostajuci.sum()

            if nedostaje_merenja > 0 and nedostaje_merenja == ukupno_merenja:
                print(f"Lokaciji {loc} nedostaju svi podaci za atribut {atribut}. Trazim odgovarajuce grupe")
                odgovrajacuciKljucevi = [k for k in kljucevi if loc in k.split(';')]
                if not odgovrajacuciKljucevi:
                    print(f"Nema odgovarajućih grupa za lokaciju {loc} i atribut {atribut}. Preskačem.\n")
                    continue
                validni_kljucevi = []
                print("Pronadjene su sledece grupe sa navedenim korelacijama:")
                for k in odgovrajacuciKljucevi:
                    print(f"  Grupa: {k.split(';')} - Prosečna korelacija: {korelacije[k].get(atribut, 0):.4f}")
                    if korelacije[k].get(atribut, 0) is not None and not np.isnan(korelacije[k].get(atribut, 0)):
                        validni_kljucevi.append(k)
                if not validni_kljucevi:
                    print("Sve dostupne grupe imaju NaN ili None korelacije. Preskačem.\n")
                    continue
                najbolja_grupa_kljuc = max(validni_kljucevi, key=lambda k: korelacije[k].get(atribut, 0))
                if korelacije[najbolja_grupa_kljuc].get(atribut, 0) is None or np.isnan(korelacije[najbolja_grupa_kljuc].get(atribut, 0)):
                    print(f"Prosečna korelacija za grupu {najbolja_grupa_kljuc.split(';')} je NaN. Preskačem.\n")
                    continue
                grupa = najbolja_grupa_kljuc.split(';')
                print(f"Odabrao sam grupu: {grupa} sa prosečnom korelacijom {korelacije[najbolja_grupa_kljuc].get(atribut, 0):.4f}")
                if korelacije[najbolja_grupa_kljuc].get(atribut, 0) < min_corr:
                    print(f"Prosečna korelacija {korelacije[najbolja_grupa_kljuc].get(atribut, 0):.4f} je ispod praga {min_corr}. Preskačem.\n")
                    continue

                donori = [d for d in grupa if d != loc]
                if not donori:
                    continue

                if atribut in ['Cloud3pm', 'Cloud9am']:
                    df_donori = df_copy[df_copy['Location'].isin(donori)][['Date', atribut]]
                    dnevni_prosek = df_donori.groupby('Date')[atribut].mean().reset_index()
                    dnevni_prosek.rename(columns={atribut: 'Prosek_Donora'}, inplace=True)
                    temp_df = df_local.merge(dnevni_prosek, on='Date', how='left')
                    features = ['Lat', 'Lon', 'Nadmorska visina (m)', 'Prosek_Donora']
                    df_train = temp_df[(temp_df['Location'] != loc) & (temp_df['Date'] < SPLIT_DATE)].dropna(subset=features + [atribut])
                    df_pred = temp_df[(temp_df['Location'] == loc) & (temp_df[atribut].isna())]

                    if len(df_train) > 10 and not df_pred.empty:
                        X_train = df_train[features]
                        y_train = df_train[atribut]
                        X_pred = df_pred[features]
                        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
                        model.fit(X_train, y_train)
                        preds = model.predict(X_pred)
                        datumi_za_upis = df_pred['Date'].values
                        df_copy.loc[(df_copy['Location'] == loc) & (df_copy['Date'].isin(datumi_za_upis)), atribut] = preds
                        print(f"  -> Primenjen XGBoost za {loc}\n")
                        uspesanXGBoost += 1
                    else:
                        print(f"  -> Nema dovoljno validnih redova za XGBoost obuku (stanica: {loc})\n")
                        neuspesanXGBoost += 1
                else:
                    podaci_idw = df_local[['Date', 'Location', atribut]]
                    pivot = podaci_idw.pivot(index='Date', columns='Location', values=atribut)
                    if loc not in pivot.columns:
                        continue
                    datumi_fale = df_copy[maska_nedostajuci]['Date']
                    for datum in datumi_fale:
                        if datum not in pivot.index:
                            continue
                        row_data = pivot.loc[datum]
                        donori_dostupni = [d for d in donori if d in row_data.index and pd.notna(row_data[d])]
                        if not donori_dostupni:
                            continue
                        tezine, vrednosti = [], []
                        for donor in donori_dostupni:
                            par = tuple(sorted([loc, donor]))
                            dist = udaljenosti_dict.get(par, np.nan)
                            if pd.notna(dist) and dist > 0:
                                tezine.append(1.0 / dist)
                                vrednosti.append(row_data[donor])
                        if tezine and np.sum(tezine) > 0:
                            procenjena_vrednost = np.average(vrednosti, weights=tezine)
                            df_copy.loc[(df_copy['Location'] == loc) & (df_copy['Date'] == datum), atribut] = procenjena_vrednost
                    print(f"  -> Primenjen IDW za {loc}\n")

    print("\nUspesan XGBoost popunjavanja:", uspesanXGBoost)
    print("Neuspesan XGBoost popunjavanja:", neuspesanXGBoost)
    return df_copy


@task(task_id="07a_imputacija_potpuno_nedostajucih", execution_timeout=timedelta(seconds=960), retries=1)
def imputacija_potpuno_nedostajucih():
    data = loadData("backups/weatherAusAfter9_2_6.csv")

    locations = data['Location'].unique()
    lokacije_sa_praznim_kolonama = []
    nedostajuci_atributi = tuple()
    prazniElementiDict = {}
    for loc in locations:
        subset = data[data['Location'] == loc]
        prazne_kolone = [col for col in data.columns if subset[col].isna().all()]
        if prazne_kolone:
            nedostajuci_atributi += tuple(prazne_kolone)
            lokacije_sa_praznim_kolonama.append(loc)
            print(f"Lokacija: {loc} ima prazne kolone: {prazne_kolone}")
            prazniElementiDict[loc] = prazne_kolone
    print("\nLista lokacija sa potpuno praznim kolonama za neke atribute:")
    print(lokacije_sa_praznim_kolonama)
    print("\nPotpuno nedostajući atributi u pojedinim meteorološkim stanicama:")
    print(set(nedostajuci_atributi))
    print(prazniElementiDict)

    _generisi_razdaljine('GeoPodaci/stanice.csv', 'GeoPodaci/udaljenost_stanica.csv')

    trazeni_pragovi = [0, 50, 100, 200, 300]
    pronadjene_grupe = _pronadji_grupe_lokacija("GeoPodaci/udaljenost_stanica.csv", trazeni_pragovi)

    podaci_za_csv = []
    for prag in trazeni_pragovi:
        grupe_za_prag = pronadjene_grupe.get(prag, [])
        for i, grupa in enumerate(grupe_za_prag, 1):
            podaci_za_csv.append({'Prag_km': prag, 'Grupa_ID': i, 'Broj_clanova': len(grupa), 'Clanovi': ', '.join(grupa)})
    if podaci_za_csv:
        pd.DataFrame(podaci_za_csv).to_csv("StationClasters/pronadjene_grupe.csv", index=False, encoding='utf-8')

    atributi_za_analizu = ['Evaporation', 'Sunshine', "Pressure9am", "Pressure3pm", "Cloud3pm", "Cloud9am"]
    prazniElementiDictFiltered = {k: [a for a in v if a in atributi_za_analizu] for k, v in prazniElementiDict.items() if any(attr in v for attr in atributi_za_analizu)}

    kor = {}
    for razdaljina in trazeni_pragovi:
        kor[str(razdaljina)] = _generisi_matricu_korelacija_po_grupama_i_atributima(data, razdaljina, nedostajuci_atributi, pronadjene_grupe)

    for atribut in atributi_za_analizu:
        nedostajuci_count_pocetak = data[atribut].isna().sum()
        print(f"\n--- Obrada atributa: {atribut} ---")

        print("[50 km] Započinjem popunjavanje...")
        data = _popuni_po_grupama(data, pronadjene_grupe.get(50, []), atribut, 0.95, kor.get("50", {}))

        preostalo_nedostajucih = data[atribut].isna().sum()
        if preostalo_nedostajucih > 0:
            print(f"\n[info] Ostalo je {preostalo_nedostajucih} nedostajućih vrednosti za {atribut}.")
            print("[100 km] Pokrećem prošireno popunjavanje...")
            data = _popuni_po_grupama(data, pronadjene_grupe.get(100, []), atribut, 0.95, kor.get("100", {}))
            konacno_nedostaje = data[atribut].isna().sum()
            print(f"\nNa pocetku bilo je {nedostajuci_count_pocetak} nedostajućih vrednosti za {atribut}.")
            print(f"Nakon 50km bilo je {preostalo_nedostajucih} nedostajućih vrednosti za {atribut}.")
            if konacno_nedostaje > 0:
                print(f"I nakon 100 km ostalo je {konacno_nedostaje} nedostajućih vrednosti.")
            else:
                print("Sve preostale vrednosti su uspešno popunjene sa 100 km.")
        else:
            print(f"\nSve vrednosti za {atribut} su uspešno popunjene u krugu od 50 km.")
        write_log(data, "Popunjen atribut " + atribut, None)

    print("Imputacija je gotova")
    data = createVaribales(data)
    write_log(data, "Imputirani svi potpuno nedostajuci atributi", "weatherAusAfter11_1_4.csv")
    # dijagnosticki kreirajAnalizu() poziv (RF feature importance/mutual info, isti
    # kod kao iskljucena sveska 06) namerno izostavljen - vidi napomenu na vrhu fajla.


# ---------------------------------------------------------------------------
# 07b_imputacija_nasumicno_nedostajucih
# ---------------------------------------------------------------------------

_CORE_FEATURES = [
    "Nadmorska visina (m)", "Inv_Dist_Sever", "Inv_Dist_Jug", "Inv_Dist_Istok", "Inv_Dist_Zapad",
    "klima_Grassland", "klima_Subtropical", "klima_Temperate", "klima_Tropical",
]


def _imputeData(target_attr, df, prediktori, stvarnaImputacija=False, backupName=None):
    df = df.loc[:, ~df.columns.duplicated()]
    broj_nedostajucih = df[target_attr].isna().sum()
    if broj_nedostajucih == 0:
        print(f"Kolona '{target_attr}' nema nedostajućih vrednosti. Imputacija nije potrebna.")
        return df
    proc_nedostajucih = (df[target_attr].isna().sum() / len(df)) * 100

    prediktori = list(dict.fromkeys(prediktori + _CORE_FEATURES))
    poznati_df = df[df[target_attr].notna() & (df['Date'] < SPLIT_DATE)].copy()
    nedostajuci_df = df[df[target_attr].isna()].copy()

    imputer = SimpleImputer(strategy='median')
    X_poznati = imputer.fit_transform(poznati_df[prediktori])
    y_poznati = poznati_df[target_attr].values

    metrics = {'mae': [], 'rmse': [], 'r2': [], 'adj_r2': []}
    for i in range(2):
        X_train, X_test, y_train, y_test = train_test_split(X_poznati, y_poznati, test_size=0.1, random_state=i)
        rf = RandomForestRegressor(n_estimators=30, random_state=i, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        n, p = len(y_test), X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        metrics['mae'].append(mae)
        metrics['rmse'].append(rmse)
        metrics['r2'].append(r2)
        metrics['adj_r2'].append(adj_r2)

    avg_mae, avg_rmse, avg_r2, avg_adj_r2 = (np.mean(metrics[k]) for k in ('mae', 'rmse', 'r2', 'adj_r2'))
    spisak_ocena = f"MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, R2={avg_r2:.4f}, Adj R2={avg_adj_r2:.4f}, postoji {proc_nedostajucih:.2f}% nedostjucih podaataka u koloni"
    poruka = f"imputirani nedostajući podaci za \natribut {target_attr}, \nmetoda RF, \nprediktori: {', '.join(prediktori)}, \nstatističke ocene: {spisak_ocena}"
    print("--- Evaluacija završena ---")
    print(poruka)
    print("---------------------------\n")

    if stvarnaImputacija:
        rf_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_final.fit(X_poznati, y_poznati)
        X_nedostajuci = imputer.transform(nedostajuci_df[prediktori])
        imputirane_vrednosti = rf_final.predict(X_nedostajuci)
        noviDF = df.copy()
        noviDF.loc[noviDF[target_attr].isna(), target_attr] = imputirane_vrednosti
        proc_nedostajucih_sad = (noviDF[target_attr].isna().sum() / len(noviDF)) * 100
        print(f"Stvarna imputacija zavrsena, u koloni sada ima {proc_nedostajucih_sad}% nedostajucih podataka")
        write_log(noviDF, poruka, backupName)
        noviDF = createVaribales(noviDF)
        return noviDF
    return df


def _imputeData_IDW(target_attr, df, df_dist, max_dist, stvarnaImputacija=False, backupName=None):
    proc_nedostajucih = df[target_attr].isna().sum()

    print("Pravljenje rečnika prostornih udaljenosti...")
    dist_dict = {}
    for _, row in df_dist.iterrows():
        s1, s2, d = row['stanica_1'], row['stanica_2'], row['razdaljina_km']
        if d <= max_dist:
            dist_dict.setdefault(s1, {})[s2] = d
            dist_dict.setdefault(s2, {})[s1] = d

    def dobiji_susede(stanica, datum, lookup_dict):
        susedi_info = dist_dict.get(stanica, {})
        validni_susedi = []
        for sused, dist in susedi_info.items():
            vrednost_suseda = lookup_dict.get((sused, datum))
            if vrednost_suseda is not None and pd.notnull(vrednost_suseda):
                validni_susedi.append({'value': vrednost_suseda, 'distance': dist})
        return validni_susedi

    poznati_indeksi = df[df[target_attr].notna()].index
    nedostajuci_indeksi = df[df[target_attr].isna()].index

    metrics = {'mae': [], 'rmse': [], 'r2': [], 'adj_r2': []}
    print("Započinjem evaluaciju (2 iteracije)...")
    for i in range(2):
        np.random.seed(i)
        test_size = int(0.1 * len(poznati_indeksi))
        test_indeksi = np.random.choice(poznati_indeksi, size=test_size, replace=False)
        train_indeksi = poznati_indeksi.difference(test_indeksi)
        df_train = df.loc[train_indeksi]
        train_lookup = df_train.set_index(['Location', 'Date'])[target_attr].to_dict()
        y_test, y_pred = [], []
        for idx in test_indeksi:
            red = df.loc[idx]
            susedi = dobiji_susede(red['Location'], red['Date'], train_lookup)
            if not susedi:
                continue
            vrednosti = [s['value'] for s in susedi]
            udaljenosti = [s['distance'] for s in susedi]
            tezine = [1.0 / (d + 1e-6) for d in udaljenosti]
            y_pred.append(np.average(vrednosti, weights=tezine))
            y_test.append(red[target_attr])
        if len(y_test) > 0:
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            n, p = len(y_test), 1
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
            metrics['r2'].append(r2)
            metrics['adj_r2'].append(adj_r2)

    ukupno_nedostajucih = len(nedostajuci_indeksi)
    broj_popunjivih = 0
    full_lookup = df.set_index(['Location', 'Date'])[target_attr].to_dict()
    if ukupno_nedostajucih > 0:
        for idx in nedostajuci_indeksi:
            red = df.loc[idx]
            if dobiji_susede(red['Location'], red['Date'], full_lookup):
                broj_popunjivih += 1
        proc_popunjivih = (broj_popunjivih / ukupno_nedostajucih) * 100
    else:
        proc_popunjivih = 0.0

    if metrics['mae']:
        avg_mae, avg_rmse, avg_r2, avg_adj_r2 = (np.mean(metrics[k]) for k in ('mae', 'rmse', 'r2', 'adj_r2'))
        spisak_ocena = (f"MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, R2={avg_r2:.4f}, Adj R2={avg_adj_r2:.4f}, "
                        f"početno nedostajućih={proc_nedostajucih:.2f} od cega je moguće popuniti={proc_popunjivih:.2f}%")
    else:
        spisak_ocena = "Evaluacija nemoguća (nema dovoljno validnih suseda u blizini)"

    poruka = f"Imputirani nedostajući podaci \natribut: {target_attr}), \nmetoda: Ponderisani prosek (Inverzna udaljenost, max_dist={max_dist}), \nocene: {spisak_ocena}"
    print("--- Evaluacija završena ---")
    print(poruka)
    print("---------------------------\n")

    if stvarnaImputacija:
        noviDF = df.copy()
        print("Započinjem stvarnu imputaciju podataka...")
        for idx in nedostajuci_indeksi:
            red = df.loc[idx]
            susedi = dobiji_susede(red['Location'], red['Date'], full_lookup)
            if susedi:
                vrednosti = [s['value'] for s in susedi]
                udaljenosti = [s['distance'] for s in susedi]
                tezine = [1.0 / (d + 1e-6) for d in udaljenosti]
                noviDF.at[idx, target_attr] = np.average(vrednosti, weights=tezine)
        write_log(noviDF, poruka, backupName)
        noviDF = createVaribales(noviDF)
        return noviDF
    return df


def _izracunaj_vlaznost(temp, tacka_rose):
    e = np.exp((17.625 * tacka_rose) / (243.04 + tacka_rose))
    e_s = np.exp((17.625 * temp) / (243.04 + temp))
    rh = 100 * (e / e_s)
    return np.clip(rh, 0, 100)


def _procesuiraj_vreme(df, vreme):
    temp_kol = f'Temp{vreme}'
    rose_kol = f'TackaRose{vreme}'
    hum_kol = f'Humidity{vreme}'

    print(f"\nIzveštaj za termin: {vreme.upper()}")
    df[temp_kol] = pd.to_numeric(df[temp_kol], errors='coerce')
    df[rose_kol] = pd.to_numeric(df[rose_kol], errors='coerce')
    df[hum_kol] = pd.to_numeric(df[hum_kol], errors='coerce')
    imputacija_maska = df[hum_kol].isna() & df[temp_kol].notna() & df[rose_kol].notna()
    broj_za_popunu = imputacija_maska.sum()
    print(f"[Imputacija] Identifikovano za popunjavanje: {broj_za_popunu} redova")
    if broj_za_popunu > 0:
        df.loc[imputacija_maska, hum_kol] = _izracunaj_vlaznost(df.loc[imputacija_maska, temp_kol], df.loc[imputacija_maska, rose_kol])
        print(" -> Popunjavanje uspešno izvršeno!")

    preostalo_maska = df[hum_kol].isna()
    broj_preostalih = preostalo_maska.sum()
    preostalo_ima_rose = df.loc[preostalo_maska, rose_kol].notna().sum()
    print(f"[Izveštaj] Neuspešno popunjeno (konačan broj nedostajućih): {broj_preostalih} redova")
    if broj_preostalih > 0:
        print(f" -> Od tih {broj_preostalih}, podatak o tački rose postoji u {preostalo_ima_rose} redova.")
    return df


@task(task_id="07b_imputacija_nasumicno_nedostajucih", execution_timeout=timedelta(seconds=2460), retries=1)
def imputacija_nasumicno_nedostajucih():
    data = loadData("backups/weatherAusAfter11_1_4.csv")
    originalDF = loadData("InputData/weatherAUS.csv")
    originalCols = [x for x in data.columns if x in originalDF.columns]
    print(originalCols)
    print(data.isna().sum().sort_values(ascending=True))

    POTREBAN_BROJ_PODATAKA_U_REDU = 15
    data = data.dropna(subset=originalCols, thresh=POTREBAN_BROJ_PODATAKA_U_REDU)
    write_log(data, "Brisanje vrsta sa velikim brojem nedostajucih vrednosti", "WatherAus_After_11_2_1.csv")

    data = loadData("backups/WatherAus_After_11_2_1.csv")
    data = _imputeData("MinTemp", data, ['Temp3pm_rolling_mean_3', 'Pressure9am', 'Temp9am',
                                          'Temp3pm', 'Year_Scaled', 'DayOfYear_Sin', 'DayOfYear_Cos',
                                          'WindSpeed9am', 'Humidity9am'], True, "WeatherAus_After_11_2_1_1.csv")

    data = _imputeData("MaxTemp", data, ['Temp3pm_rolling_mean_3', 'Pressure3pm', 'Temp9am',
                                          'Temp3pm', 'Year_Scaled', 'DayOfYear_Sin',
                                          'DayOfYear_Cos', 'WindSpeed3pm', 'Humidity3pm'], True,
                        "WeatherAus_After_11_2_1_2.csv")

    data = _imputeData("Temp9am", data, ['Temp3pm_rolling_mean_3', 'Pressure3pm', 'MinTemp',
                                          'Temp3pm', 'Year_Scaled', 'DayOfYear_Sin', 'DayOfYear_Cos',
                                          'WindSpeed3pm', 'Humidity3pm'], True,
                        "WeatherAus_After_11_2_1_3.csv")

    data = _imputeData("Temp3pm", data, ['Temp3pm_rolling_mean_3', 'Pressure3pm', 'MinTemp',
                                          'Temp9am', 'Year_Scaled', 'DayOfYear_Sin', 'DayOfYear_Cos',
                                          'WindSpeed3pm', 'Humidity3pm'], True,
                        "WeatherAus_After_11_2_1_4.csv")

    razdaljine_df = pd.read_csv("GeoPodaci/udaljenost_stanica.csv")

    data = loadData("backups/WeatherAus_After_11_2_1_4.csv")
    data = _imputeData_IDW("Pressure9am", data, razdaljine_df, 150, True, "WeatherAus_After_11_2_1_5a.csv")

    data = _imputeData("Pressure9am", data,
                        ["Pressure3pm", 'Temp9am', 'Temp3pm', 'Year_Scaled',
                         'DayOfYear_Sin', 'DayOfYear_Cos', 'Pressure9am_pre_1_dana',
                         'Pressure3pm_pre_1_dana', 'Temp9am_Min_diff', 'Max_Temp3pm_diff',
                         'WindSpeed3pm', 'WindDir3pm_sin', 'WindDir3pm_cos',
                         'WindDir9am_sin', 'WindDir9am_cos', 'WindSpeed3pm_rolling_mean_3'],
                        True, "WeatherAus_After_11_2_1_5b.csv")

    data = _imputeData_IDW("Pressure3pm", data, razdaljine_df, 150, True, "WeatherAus_After_11_2_1_6a.csv")

    data = _imputeData("Pressure3pm", data,
                        ['Pressure9am', 'Temp9am', 'Temp3pm', 'Year_Scaled',
                         'DayOfYear_Sin', 'DayOfYear_Cos', 'Pressure9am_pre_1_dana',
                         'Pressure3pm_pre_1_dana', 'Temp9am_Min_diff',
                         'Max_Temp3pm_diff', 'WindSpeed3pm', 'WindDir3pm_sin',
                         'WindDir3pm_cos', 'WindDir9am_sin', 'WindDir9am_cos',
                         'WindSpeed3pm_rolling_mean_3'], True,
                        "WeatherAus_After_11_2_1_6b.csv")

    data = _procesuiraj_vreme(data, '9am')
    data = _procesuiraj_vreme(data, '3pm')

    data = _imputeData("Humidity9am", data,
                        ['Temp3pm', 'Temp9am', "Temp9am_Min_diff", "Max_Temp3pm_diff",
                         "Max_Min_Temp_Diff", 'Rainfall', 'WindSpeed9am',
                         "Humidity3pm", "Lon"], True, None)

    data = _imputeData("Humidity3pm", data,
                        ["Humidity9am", "Pressure3pm", "Temp3pm", "Max_Min_Temp_Diff",
                         "Temp3pm_rolling_mean_3", "Temp9am_Min_diff", "Max_Temp3pm_diff",
                         "Pressure3pm_pre_1_dana", "Inv_Dist_Jug"],
                        True, "WeatherAus_After_11_2_3_7.csv")

    # NAPOMENA: checkpoint ispod ("WeatherAus_After_11_2_3_8.csv") je poslednji koji
    # naredna faza (08_finalne_predikcije, DATASET_PATH) stvarno cita. Dijagnosticki
    # zavrsni deo originalne sveske (R^2 poredjenje RF-a naspram medijane, histogrami,
    # kreirajAnalizu()) je namerno izostavljen - vidi napomenu na vrhu fajla.
    data = _imputeData("Sunshine", data,
                        ['Cloud9am', 'Cloud3pm', 'Humidity9am', 'Humidity3pm',
                         'Rainfall', 'MaxTemp', 'Temp3pm', 'Pressure9am', 'Pressure3pm',
                         'Max_Min_Temp_Diff', 'DayOfYear_Sin', 'DayOfYear_Cos',
                         'WindGustSpeed'], True, "WeatherAus_After_11_2_3_8.csv")


# ---------------------------------------------------------------------------
# 08_finalne_predikcije
# ---------------------------------------------------------------------------

def _optimizuj_prag_odlucivanja(y_test, y_prob):
    pragovi = np.arange(0.1, 0.82, 0.02)
    f1_skorovi = []
    for prag in pragovi:
        y_pred_custom = (y_prob >= prag).astype(int)
        f1_skorovi.append(f1_score(y_test, y_pred_custom))
    najbolji_indeks = np.argmax(f1_skorovi)
    return pragovi[najbolji_indeks]


class _RainPredictor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def ucitaj_podatke(self):
        self.df = pd.read_csv(self.filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(by='Date')
        self.df.dropna(subset=['RainTomorrow'], inplace=True)
        self.df.drop(columns=["Cloud9am", "RainToday", "Cloud3pm", "Evaporation", "Location",
                               "TackaRose9am_missing", "TackaRose3pm_missing", "TackaRose9am", "TackaRose3pm"],
                      inplace=True)
        for col in ("Unnamed: 0.2", "Unnamed: 0", "Unnamed: 0.1"):
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)

    def _pretprocesiraj_skup(self, data: pd.DataFrame):
        df_processed = data.copy()
        X = df_processed.drop(columns=["RainTomorrow"])
        y = df_processed['RainTomorrow'].map({'No': 0, 'Yes': 1})
        return X, y

    def _napravi_model(self, model_type: str, use_class_weight: bool):
        if model_type == 'xgb':
            scale_pos_weight = 1.5 if use_class_weight else 1.0
            return XGBClassifier(
                n_estimators=1625, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                gamma=0.2, reg_alpha=1.0, reg_lambda=2.0,
                random_state=42, n_jobs=-1, eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
            )
        elif model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                class_weight=('balanced' if use_class_weight else None),
                min_samples_leaf=5,
            )
        else:
            return DecisionTreeClassifier(
                random_state=42,
                class_weight=('balanced' if use_class_weight else None),
                min_samples_leaf=5,
                max_depth=10,
            )

    def treniraj_i_evaluiraj(self, ime_skupa: str, atributi, model_type: str = 'xgb', use_class_weight: bool = True, prag_fiksni: float = None):
        self.atributi = atributi
        with mlflow.start_run(run_name="unified_noleak_" + ime_skupa):
            print(f"\nProcesiranje i modelovanje za: {ime_skupa}")

            X_full, y = self._pretprocesiraj_skup(self.df)
            is_train = X_full['Date'] < SPLIT_DATE
            dates = X_full['Date']
            X = X_full[self.atributi]
            X_train, X_test = X[is_train], X[~is_train]
            y_train, y_test = y[is_train], y[~is_train]
            num_cols = X.select_dtypes(include=['float64', 'int64']).columns

            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_cols)])
            pipeline = Pipeline(steps=[('preprocesiranje', preprocessor), ('model', self._napravi_model(model_type, use_class_weight))])
            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            if prag_fiksni is not None:
                optimalni_prag = prag_fiksni
                print(f"Koristi se fiksan prag: {optimalni_prag:.2f} (bez F1-optimizacije)")
            else:
                train_dates = dates[is_train]
                val_cutoff = train_dates.quantile(0.85)
                is_fit = is_train & (dates < val_cutoff)
                is_val = is_train & (dates >= val_cutoff)

                X_fit, y_fit = X[is_fit], y[is_fit]
                X_val, y_val = X[is_val], y[is_val]

                pipeline_fit = Pipeline(steps=[
                    ('preprocesiranje', ColumnTransformer(transformers=[('num', StandardScaler(), num_cols)])),
                    ('model', self._napravi_model(model_type, use_class_weight)),
                ])
                pipeline_fit.fit(X_fit, y_fit)
                y_prob_val = pipeline_fit.predict_proba(X_val)[:, 1]
                optimalni_prag = _optimizuj_prag_odlucivanja(y_val, y_prob_val)
                print(f"Prag odredjen naslepo na internoj validaciji ({len(X_val)} redova, "
                      f"nikad video test skup): {optimalni_prag:.2f}")

            mlflow.log_params({"optimalni_prag": optimalni_prag, "use_class_weight": use_class_weight})
            y_pred_optimalno = (y_prob >= optimalni_prag).astype(int)

            print(f"\nFinalni izveštaj klasifikacije sa pragom od {optimalni_prag:.2f}:")
            print(classification_report(y_test, y_pred_optimalno))

            acc = accuracy_score(y_test, y_pred_optimalno)
            roc_auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred_optimalno)
            precision = precision_score(y_test, y_pred_optimalno)
            recall = recall_score(y_test, y_pred_optimalno)
            mlflow.log_metric("acc", acc)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            print(f"Tačnost (Accuracy): {acc:.4f}")
            print(f"ROC AUC Skor: {roc_auc:.4f}")
            print(f"F1 Skor (Klasa 1):  {f1:.4f}")
            print(f"Preciznost (Klasa 1): {precision:.4f}")
            print(f"Odziv (Klasa 1): {recall:.4f}")
            mlflow.sklearn.log_model(pipeline, name="moj_model " + ime_skupa, serialization_format="cloudpickle")


@task(task_id="08_finalne_predikcije", execution_timeout=timedelta(seconds=1860), retries=1)
def finalne_predikcije():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("WeatherAus proba 1")

    DATASET_PATH = "backups/WeatherAus_After_11_2_3_8.csv"
    predictor = _RainPredictor(filepath=DATASET_PATH)
    predictor.ucitaj_podatke()

    ATRIBUTI_UNIFIED = ['Rainfall', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
                         'Pressure9am', 'Pressure3pm', 'Max_Min_Temp_Diff', 'Temp3pm_rolling_mean_3',
                         'Pressure3pm_pre_1_dana']

    predictor.treniraj_i_evaluiraj(ime_skupa='stablo_tezinsko_f1', atributi=ATRIBUTI_UNIFIED,
                                    model_type='dt', use_class_weight=True, prag_fiksni=None)
    predictor.treniraj_i_evaluiraj(ime_skupa='suma_tezinsko_f1', atributi=ATRIBUTI_UNIFIED,
                                    model_type='rf', use_class_weight=True, prag_fiksni=None)
    predictor.treniraj_i_evaluiraj(ime_skupa='xgboost_tezinsko_f1', atributi=ATRIBUTI_UNIFIED,
                                    model_type='xgb', use_class_weight=True, prag_fiksni=None)
    predictor.treniraj_i_evaluiraj(ime_skupa='xgboost_osnovno_05', atributi=ATRIBUTI_UNIFIED,
                                    model_type='xgb', use_class_weight=False, prag_fiksni=0.50)


# ---------------------------------------------------------------------------
# 09_produkcija_modela
# ---------------------------------------------------------------------------

@task(task_id="09_produkcija_modela", execution_timeout=timedelta(seconds=360), retries=1)
def produkcija_modela():
    from mlflow import MlflowClient
    import requests

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

    REGISTROVANI_MODEL_NAME = "WeatherAusRainModel"
    CILJANI_RUN_NAME = "unified_noleak_xgboost_tezinsko_f1"

    experiment = client.get_experiment_by_name("WeatherAus proba 1")
    kandidati = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{CILJANI_RUN_NAME}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not kandidati:
        raise RuntimeError(
            f"Nije pronadjen nijedan MLflow run sa imenom '{CILJANI_RUN_NAME}' u eksperimentu "
            f"'WeatherAus proba 1' - da li je task finalne_predikcije uspesno izvrsen?"
        )

    pobednicki_run = client.get_run(kandidati[0].info.run_id)
    model_id = pobednicki_run.outputs.model_outputs[0].model_id
    optimalni_prag = float(pobednicki_run.data.params["optimalni_prag"])

    registrovan = mlflow.register_model(f"models:/{model_id}", REGISTROVANI_MODEL_NAME)
    client.set_registered_model_alias(REGISTROVANI_MODEL_NAME, "champion", registrovan.version)
    client.set_model_version_tag(REGISTROVANI_MODEL_NAME, registrovan.version, "optimalni_prag", str(optimalni_prag))

    print(f"Registrovan '{REGISTROVANI_MODEL_NAME}' v{registrovan.version} (run {pobednicki_run.info.run_id}) sa aliasom 'champion'")
    print(f"Optimalni prag odlucivanja koji API koristi: {optimalni_prag}")

    API_RELOAD_URL = os.environ.get("API_RELOAD_URL", "http://api:8000/reload")
    try:
        odgovor = requests.post(API_RELOAD_URL, timeout=15)
        odgovor.raise_for_status()
        print(f"API osvezen: {odgovor.json()}")
    except Exception as e:
        print(f"[upozorenje] Nisam mogao da osvezim API na {API_RELOAD_URL} ({e}). "
              f"Ako je 'api' servis gore, osvezi rucno: curl -X POST {API_RELOAD_URL}")
