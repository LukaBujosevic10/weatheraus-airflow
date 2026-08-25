"""
Zivo dohvatanje meteoroloskih podataka sa Open-Meteo (besplatno, bez API kljuca:
https://open-meteo.com) i preracunavanje istog feature-seta koji ocekuje "live"
model (10 od originalnih 12 atributa - bez Sused_RainToday_pct/Sused_Pressure3pm_avg,
koji bi zahtevali live podatke sa SVIH susednih stanica istovremeno, ne samo jedne).

Vazna napomena o datumu: dnevni agregati (Rainfall/Sunshine/...) za DANASNJI dan su
nepotpuni dok dan traje (kisa/sunce se jos akumuliraju). Zato se kao osnova UVEK
koristi poslednji POTPUNO zavrsen dan (juce), isto onako kako trening podaci uvek
predstavljaju zavrsen dan. Vracen "date" u odgovoru je taj dan, a predikcija je za
dan POSLE njega (u praksi: danas/sutra, zavisno od doba dana kad se poziva API).
"""
import requests

ATRIBUTI_LIVE = [
    "Rainfall", "Sunshine", "WindGustSpeed", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Max_Min_Temp_Diff", "Temp3pm_rolling_mean_3",
    "Pressure3pm_pre_1_dana",
]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _sat_iz_niza(vremena, vrednosti, datum, sat):
    trazeno = f"{datum}T{sat:02d}:00"
    try:
        idx = vremena.index(trazeno)
    except ValueError:
        return None
    return vrednosti[idx]


def izracunaj_live_atribute(lat: float, lon: float) -> tuple[dict, str, list[str]]:
    """Vraca (vrednosti_atributa, datum_baznog_dana, lista_nedostajucih_atributa)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl",
        "daily": "precipitation_sum,sunshine_duration,wind_gusts_10m_max,temperature_2m_max,temperature_2m_min",
        "past_days": 5,
        "forecast_days": 1,
        "timezone": "auto",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    r.raise_for_status()
    podaci = r.json()

    daily = podaci["daily"]
    hourly = podaci["hourly"]
    datumi = daily["time"]

    # datumi[-1] je "sutra" (forecast_days=1), datumi[-2] je danas (nepotpun dok traje) -
    # bazni dan je poslednji ZAVRSEN dan, datumi[-3].
    bazni_datum = datumi[-3]
    juce_datum = datumi[-4]
    pre_3_dana = [datumi[-5], datumi[-4], datumi[-3]]

    idx_bazni = datumi.index(bazni_datum)

    humidity_9am = _sat_iz_niza(hourly["time"], hourly["relative_humidity_2m"], bazni_datum, 9)
    humidity_3pm = _sat_iz_niza(hourly["time"], hourly["relative_humidity_2m"], bazni_datum, 15)
    pressure_9am = _sat_iz_niza(hourly["time"], hourly["pressure_msl"], bazni_datum, 9)
    pressure_3pm = _sat_iz_niza(hourly["time"], hourly["pressure_msl"], bazni_datum, 15)
    pressure_3pm_juce = _sat_iz_niza(hourly["time"], hourly["pressure_msl"], juce_datum, 15)

    temp3pm_vrednosti = [
        v for v in (
            _sat_iz_niza(hourly["time"], hourly["temperature_2m"], d, 15) for d in pre_3_dana
        ) if v is not None
    ]
    temp3pm_rolling_mean_3 = (
        round(sum(temp3pm_vrednosti) / len(temp3pm_vrednosti), 2) if temp3pm_vrednosti else None
    )

    rainfall = daily["precipitation_sum"][idx_bazni]
    sunshine_sekunde = daily["sunshine_duration"][idx_bazni]
    sunshine = round(sunshine_sekunde / 3600.0, 1) if sunshine_sekunde is not None else None
    wind_gust = daily["wind_gusts_10m_max"][idx_bazni]
    temp_max = daily["temperature_2m_max"][idx_bazni]
    temp_min = daily["temperature_2m_min"][idx_bazni]
    max_min_diff = round(temp_max - temp_min, 2) if (temp_max is not None and temp_min is not None) else None

    vrednosti = {
        "Rainfall": rainfall,
        "Sunshine": sunshine,
        "WindGustSpeed": wind_gust,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Max_Min_Temp_Diff": max_min_diff,
        "Temp3pm_rolling_mean_3": temp3pm_rolling_mean_3,
        "Pressure3pm_pre_1_dana": pressure_3pm_juce,
    }
    nedostaju = [k for k, v in vrednosti.items() if v is None]
    return vrednosti, bazni_datum, nedostaju
