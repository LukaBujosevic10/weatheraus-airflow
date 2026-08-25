# WeatherAus - Airflow

Ovaj projekat je produkciona/orkestraciona nadogradnja projekta [**Predikcija kiše u Australiji**](../merging/README.md), rađenog u okviru predmeta Uvod u nauku o podacima.

## Opis projekta

Cilj projekta je praktična primena Apache Airflow-a i automatizacija celokupnog data science ciklusa - od obrade meteoroloških podataka i detekcije anomalija, preko inženjeringa obeležja i popunjavanja nedostajućih vrednosti, do treniranja modela i serviranja predikcija da li će sutra padati kiša.

Korišćene tehnologije:

- **Apache Airflow** - orkestracija i automatizacija pipeline-a
- **MLflow** - praćenje eksperimenata (metrike po modelu) i model registry
- **FastAPI** - REST API za serviranje predikcija
- **Streamlit** - interaktivna mapa meteoroloških stanica

## Sadržaj

1. [Šta je Airflow i čemu služi?](#1-šta-je-airflow-i-čemu-služi)
2. [Kako smo ga mi iskoristili?](#2-kako-smo-mi-iskoristili-airflow)
3. [Kako pokrenuti ceo projekat?](#3-kako-pokrenuti-ceo-projekat)

## 1. Šta je Airflow i čemu služi?

Apache Airflow je open-source alat za kreiranje, zakazivanje i praćenje pipeline-a. Umesto ručnog, korak-po-korak pokretanja skripti (ili u našem slučaju - svezaka), Airflow omogućava da se ceo tok obrade podataka definiše kroz kod (Python), kao **DAG** (Directed Acyclic Graph, usmeren aciklički graf koraka koji zavise jedan od drugog).

(Sve što Airflow radi teoretski bi moglo i ručno, samo mnogo sporije, bez uvida u to šta je prošlo/palo, i bez ikakvog automatskog ponavljanja.)

Ključne prednosti:

- **Programabilnost** - ceo tok se piše u Python-u, potpuna fleksibilnost.
- **Vizuelizacija i UI** - pregledan web interfejs za praćenje izvršavanja, logova i grešaka u realnom vremenu (kod nas na http://localhost:8081).
- **Otpornost na greške** - automatsko ponavljanje neuspelih koraka, sa posebnim timeout-om po koraku.
- **Skalabilnost** - isti alat pokriva i jednostavan dnevni posao i hiljade složenih koraka u velikim sistemima.

## 2. Kako smo mi iskoristili Airflow

### Zamisao

Skup podataka koji koristimo (Rain in Australia, 145.000+ dnevnih merenja sa 49 stanica) je istorijski i fiksan, ali ceo pipeline - od sirovih podataka do registrovanog modela - je namerno napravljen kao da se svakog dana može ponovo pokrenuti nad osveženim podacima: ista sekvenca koraka, iste provere, isti model registry. Ovo je vežba automatizacije celog data science ciklusa od kraja do kraja, ne samo treniranja jednog modela.

### Struktura zadataka (Tasks)

Za razliku od jednostavnijih pipeline-ova (predobrada → trening → evaluacija, tri koraka), naš pipeline ima **8 međusobno zavisnih taskova**, jer originalna analiza ima znatno više faza sa sopstvenim checkpoint-ima:

1. `01_ucitavanje_i_sistemske_greske` - učitavanje i ispravka fizički nemogućih merenja (npr. `MinTemp > MaxTemp`)
2. `04_detekcija_anomalija` - šestostruka detekcija anomalija (MAD Z-skor, IQR, Mahalanobis, Isolation Forest, LOF, KNN), fit isključivo na trening delu
3. `05a_priprema_vremenskih_i_diferencijalnih_atributa` - ciklično kodiranje vremena/vetra, diferencijalni i lag atributi
4. `05b_geo_i_dopunski_atributi` - geografske/klimatske karakteristike stanica, tačka rose, indikatori nedostajućih vrednosti
5. `07a_imputacija_potpuno_nedostajucih` - prostorna imputacija za atribute koji potpuno nedostaju na nekim stanicama
6. `07b_imputacija_nasumicno_nedostajucih` - Random Forest/IDW imputacija za nasumično nedostajuće vrednosti (najskuplji korak)
7. `08_finalne_predikcije` - treniranje i evaluacija 4 varijante modela (Decision Tree, Random Forest, XGBoost x2), sve praćeno kroz MLflow
8. `09_produkcija_modela` - registracija pobedničkog modela u MLflow Model Registry i osvežavanje FastAPI servisa

Svaki task čita CSV checkpoint koji je prethodni upravo napisao - tok podataka je strogo linearan, bez grananja.

**Implementacija:** funkcije koje izvršavaju ove zadatke napisane su u [`utils/tasks.py`](utils/tasks.py), uz zajedničke konstante/pomoćne funkcije u [`utils/common.py`](utils/common.py). Ovo nije pisano od nule - izvučeno je iz svezaka u [`../merging/notebooks/`](../merging/notebooks/) (ista logika, bez izmena), koje ostaju netaknute u tom odvojenom projektu i dalje služe kao glavna dokumentacija analize i obrazloženja odluka. Sami podaci (`InputData/`, `GeoPodaci/`, keširani proračuni) su kopirani u [`data/`](data/README.md) ovog projekta - pipeline je samostalan, ne čita `../merging` u radu. Namerno su izostavljeni koraci koji su čisto dijagnostički i ne pišu ništa što naredna faza koristi (EDA, analiza značaja atributa...) - detaljno obrazloženje u [`AIRFLOW_SETUP.md`](AIRFLOW_SETUP.md).

### Orkestracija pomoću Airflow DAG-a

Kada su task funkcije definisane, ostalo je samo da se povežu pravim redosledom - ovo je urađeno u [`dags/weatheraus_pipeline_dag.py`](dags/weatheraus_pipeline_dag.py), lančano (`>>`), tačno redosledom iz liste iznad. DAG je podešen na ručno pokretanje (`schedule=None`) jer je skup podataka istorijski, ne pristižu nove dnevne mere - lako se menja u npr. `schedule="@weekly"` ako se poveže sa pravim izvorom svežih podataka.

### Upravljanje i praćenje preko Airflow Web UI-ja

Na adresi http://localhost:8081 (korisničko ime/lozinka `admin`/`admin`) dostupno je:

- pregled DAG-a `weatheraus_pipeline` i njegovog statusa (aktivan/neaktivan)
- dijagram zavisnosti (Graph View) koji vizuelno prikazuje redosled od 8 taskova
- ručno pokretanje pipeline-a (Trigger DAG)
- status svakog taska pojedinačno, uživo, sa direktnim pristupom logovima za debagovanje
- istorija prethodnih pokretanja (trajanje po tasku, koji je pao i zašto)

### Povezivanje sa MLflow, FastAPI i Streamlit aplikacijom

- **Praćenje eksperimenata i model registry (MLflow):** task `08_finalne_predikcije` loguje metrike (accuracy, ROC-AUC, F1, preciznost, odziv) za sve 4 varijante modela u MLflow (http://localhost:5000); task `09_produkcija_modela` registruje pobednički model (XGBoost sa optimizovanim pragom) pod aliasom `champion`.
- **Serviranje predikcija (FastAPI):** servis `app/api` učitava `champion` model direktno iz MLflow registry-ja i servira predikcije - za razliku od pipeline-a koji radi nad istorijskim podacima, API hvata **stvarne, uživo podatke** današnjeg dana sa australijskih meteoroloških stanica za predikciju. Poslednji korak DAG-a (`09`) automatski obaveštava API da je nov model spreman, bez potrebe za restartom kontejnera.
- **Frontend (Streamlit):** aplikacija `app/ui` (http://localhost:8501) prikazuje interaktivnu mapu meteoroloških stanica - klik na stanicu poziva FastAPI i prikazuje predikciju za taj dan.

## 3. Kako pokrenuti ceo projekat?

### Preduslovi

- Docker Desktop instaliran i pokrenut - to je jedini preduslov, projekat je samostalan (`data/` već sadrži sve potrebne podatke, vidi [`data/README.md`](data/README.md))

### Koraci za pokretanje

U ovom folderu:

```powershell
docker compose up -d --build
```

Jedna komanda pokreće sve - Airflow, MLflow, FastAPI i Streamlit. Prvi put build-uje sliku (par minuta).

### Pristup aplikacijama

Nakon što se kontejneri pokrenu:

- **Airflow Web UI:** http://localhost:8081 (korisničko ime `admin`, lozinka `admin`) - uključi `weatheraus_pipeline` (toggle) i pokreni ga dugmetom Trigger DAG.
- **MLflow UI:** http://localhost:5000 - pregled eksperimenata, metrika i registrovanih modela.
- **FastAPI dokumentacija (Swagger UI):** http://localhost:8000/docs - interaktivno testiranje predikcionih ruta.
- **Streamlit mapa stanica:** http://localhost:8501 - klik na stanicu za predikciju kiše.

Kompletno uputstvo (debagovanje, poznata ograničenja, fiksna admin lozinka...) je u [`AIRFLOW_SETUP.md`](AIRFLOW_SETUP.md).
