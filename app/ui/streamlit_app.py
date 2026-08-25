"""
Streamlit UI - mapa australijskih meteoroloskih stanica; klik na stanicu poziva
FastAPI servis (app/api), koji uzivo (Open-Meteo) povuce poslednji zavrseni dan za
tu stanicu i predvidi da li sutra pada kisa.
"""
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Rain in Australia - Predikcija", layout="wide")
st.title("Rain in Australia - Kompletne stanice i predikcija")
st.caption("Izaberite meteorolosku stanicu (klikom na mapu ili iz liste) da vidite predikciju kise za sutra, na osnovu svezih podataka.")


@st.cache_data(ttl=300)
def ucitaj_stanice():
    r = requests.get(f"{API_URL}/stations", timeout=10)
    r.raise_for_status()
    return r.json()


def predvidi(location):
    r = requests.get(f"{API_URL}/predict/{location}", timeout=30)
    r.raise_for_status()
    return r.json()


try:
    stanice = ucitaj_stanice()
except Exception as e:
    st.error(f"Ne mogu da se povezem na API ({API_URL}): {e}")
    st.stop()

if not stanice:
    st.warning("API jos nema nijednu stanicu (pokreni notebooks/09_produkcija_modela.ipynb).")
    st.stop()

imena_stanica = [s["location"] for s in stanice]
if "izabrana_stanica" not in st.session_state:
    st.session_state.izabrana_stanica = imena_stanica[0]

col_mapa, col_info = st.columns([2, 1])

with col_mapa:
    mapa = folium.Map(location=[-25.0, 134.0], zoom_start=4, tiles="OpenStreetMap")
    for s in stanice:
        folium.Marker(
            [s["lat"], s["lon"]],
            tooltip=s["location"],
            popup=s["location"],
            icon=folium.Icon(color="blue", icon="cloud"),
        ).add_to(mapa)
    klik = st_folium(mapa, height=560, use_container_width=True)

if klik and klik.get("last_object_clicked_tooltip") in imena_stanica:
    st.session_state.izabrana_stanica = klik["last_object_clicked_tooltip"]

with col_info:
    izabrana = st.selectbox(
        "Stanica",
        options=imena_stanica,
        index=imena_stanica.index(st.session_state.izabrana_stanica),
    )
    st.session_state.izabrana_stanica = izabrana

    st.subheader(izabrana)
    try:
        with st.spinner("Dohvatam žive podatke..."):
            rezultat = predvidi(izabrana)
    except Exception as e:
        st.error(f"Greška pri predikciji: {e}")
        st.stop()

    prikaz_parametara = {"Date": rezultat["date"], **rezultat["features"]}
    st.table(
        {
            "Parametar": list(prikaz_parametara.keys()),
            "Vrednost": [round(v, 2) if isinstance(v, float) else v for v in prikaz_parametara.values()],
        }
    )

    st.markdown("#### PREDIKCIJA KIŠE (SUTRA)")
    boja = "#d32f2f" if rezultat["rain_tomorrow"] == "Yes" else "#2e7d32"
    st.markdown(f"<h2 style='color:{boja}'>{rezultat['rain_tomorrow']}</h2>", unsafe_allow_html=True)
    st.caption(f"Bazirano na poslednjem potpuno završenom danu ({rezultat['date']}) - predikcija je za dan posle njega.")
