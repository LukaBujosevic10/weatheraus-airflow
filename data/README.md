# data/

Ovo je **radni direktorijum pipeline-a** - montiran u Airflow kontejnere na `/opt/airflow/weatherdata` (`docker-compose.yaml`). Projekat je samostalan: više ne zavisi od sestrinskog foldera `../merging` - svi ulazni podaci koje pipeline treba da krene od nule su ovde.

## Šta je unutra

- `InputData/weatherAUS.csv`, `InputData/stations.txt` - sirovi ulazni podaci (meteorološka merenja, spisak BOM stanica). Koriste ih taskovi `01_ucitavanje_i_sistemske_greske` i `07b_imputacija_nasumicno_nedostajucih`/`05b_geo_i_dopunski_atributi`.
- `GeoPodaci/stanice.csv`, `GeoPodaci/udaljenost_stanica.csv` - koordinate/udaljenosti stanica. Ovi fajlovi se **ponovo generišu** od strane taskova `05b` (stanice.csv) i `07a` (udaljenost_stanica.csv) na svakom pokretanju pipeline-a - ono što je ovde je samo početna kopija, ništa ne mora ručno da se održava.
- `reports/tacka_rose.csv` - keširan proračun tačke rose (originalno računat iz `InputData/asos.csv`, ~160MB; taj sirovi fajl namerno NIJE kopiran ovde jer `timezonefinder` paket nije instaliran u minimalnom Airflow image-u - `05b` očekuje da ovaj keširan fajl već postoji i preskače ponovno računanje).
- `backups/weatherAusAfter9_2_5_4.csv` - keširan proračun udaljenosti stanica od okeana (originalno računat preko `geopandas`/`cartopy`, koji takođe nisu instalirani u minimalnom image-u - `05b` čita ovaj keš kao fallback).
- `backups/WeatherAus_After_11_2_3_8.csv` - finalni obrađeni dataset koji `08_finalne_predikcije` koristi za treniranje. Ovo je izlaz taska `07b` - drži se ovde kao pogodan početak ako želiš da testiraš samo `08`/`09` bez pokretanja celog pipeline-a, ali se svakako iznova piše na svakom punom pokretanju.
- `logs/`, `reports/claster_corelations/`, `AnomalyDetectionResults/`, `StationClasters/` - kreiraju se automatski (`utils/common.py`, `os.makedirs(..., exist_ok=True)`) i pune se tokom izvršavanja pipeline-a.

Ostali `backups/*.csv` checkpoint-i (medjukoraci 01→09) se pojavljuju ovde tek posle prvog pokretanja DAG-a - namerno nisu unapred seed-ovani jer ih svaki task iznova piše.

## Zašto `asos.csv` (160MB) i pun `InputData`/`backups` iz `merging` (~1GB) NISU ovde

Minimalni Airflow image (`requirements-airflow.txt`) namerno ne instalira geoprostorne pakete (`geopandas`, `cartopy`, `timezonefinder`...) - originalni kod ih hvata kroz `try/except` i pada nazad na keširane CSV-ove (`reports/tacka_rose.csv`, `backups/weatherAusAfter9_2_5_4.csv`, oba već ovde). Zato sirovi `asos.csv` nikad nije stvarno pročitan u ovom pipeline-u. Ostali `backups/*.csv` fajlovi iz `merging` (900MB+) su međurezultati koje pipeline sam iznova generiše - kopiranje bi samo trošilo prostor bez ikakve koristi.
