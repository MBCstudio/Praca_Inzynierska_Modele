# Stock Price Prediction Models

> Pretrenowane modele hybrydowe LSTM-GRU oraz model Ridge Regression do predykcji cen akcji w różnych horyzontach czasowych

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Repozytorium zawiera pretrenowane modele oraz skrypty treningowe opracowane w ramach pracy inżynierskiej na Politechnice Wrocławskiej (2025). Modele wykorzystują hybrydową architekturę LSTM-GRU oraz model Ridge Regression do prognozowania cen akcji Apple Inc. (AAPL) w horyzontach: 1, 5, 10 i 21 dni handlowych.

---

## Struktura repozytorium
```
LSTMS/
├── models/                    # Pretrenowane modele
│   ├── 1d/                   # Model 1-dniowy
│   │   ├── config.json       # Metadane i hiperparametry
│   │   ├── model_1.keras     # Model ensemble #1
│   │   ├── model_2.keras     # Model ensemble #2
│   │   ├── model_3.keras     # Model ensemble #3
│   │   └── scaler.pkl        # RobustScaler do normalizacji
│   │
│   ├── 5d/                   # Model 5-dniowy (identyczna struktura)
│   ├── 10d/                  # Model 10-dniowy (identyczna struktura)
│   └── 21d/                  # Model 21-dniowy (identyczna struktura)
│
├── graphs/                    # Wykresy ewaluacji modeli
│   ├── 1d/
│   │   ├── cumulative_returns.png
│   │   ├── predicted_vs_actual.png
│   │   └── prediction_errors.png
│   ├── 5d/
│   ├── 10d/
│   └── 21d/
│
├── Final_v2_1d.py            # Skrypt treningowy model 1d
├── Final_v2_5d.py            # Skrypt treningowy model 5d
├── Final_v2_10d.py           # Skrypt treningowy model 10d
├── Final_v2_21d.py           # Skrypt treningowy model 21d
├── new_RR_1.py               # Model Ridge Regression
│
├── bias_correction.py        # Moduł korekcji systematycznego bias
├── bias_corrections.json     # Współczynniki korekcji dla modeli
│
├── requirements.txt          # Zależności Python
├── README.md                 # Ten plik
└── LICENSE                   # Licencja MIT
```

---

## Pretrenowane modele

### Parametry architektur

| Horyzont | Typ warstw | Jednostki LSTM | Jednostki GRU | Dropout | Learning Rate | Sequence Length |
|----------|-----------|----------------|---------------|---------|---------------|-----------------|
| **1 dzień** | Bidirectional | [80, 48] | [80] | 0.25 | 0.0012 | **40 dni** |
| **5 dni** | Bidirectional | [80, 48] | [80] | 0.25 | 0.0012 | 60 dni |
| **10 dni** | Standard | [128, 96] | [128, 96] | 0.30 | 0.0008 | 60 dni |
| **21 dni** | Standard | [128, 96] | [128, 96] | 0.35 | 0.0008 | 60 dni |

### Metryki jakości (zbiór testowy)

Modele zostały wytrenowane na danych AAPL z okresu **2015-01-01 do 2025-11-18** (podział 80/20 train/test):

| Horyzont | MAE [$] | RMSE [$] | MAPE [%] | R² |
|----------|---------|----------|----------|-----|
| **1 dzień** | 2.33 | 3.05 | 1.29 | 0.81 |
| **5 dni** | 3.36 | 4.20 | 1.95 | 0.50 |
| **10 dni** | 3.65 | 4.65 | 2.09 | 0.17 |
| **21 dni** | 5.10 | 7.20 | 2.90 | -0.70 |

**Wnioski:**
- Modele osiągają bardzo dobre wyniki dla horyzontów **1-10 dni**
- Model 21-dniowy ma ograniczoną użyteczność (ujemne R²) – wymaga włączenia danych fundamentalnych

---

## Konfiguracja i uruchomienie

### Wymagania
```bash
Python 3.9+
TensorFlow 2.x
scikit-learn
NumPy, pandas
yfinance
matplotlib
```

### Instalacja
```bash
# Sklonuj repozytorium
git clone https://github.com/twoj-username/stock-prediction-models.git
cd stock-prediction-models

# Utwórz środowisko wirtualne
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# lub
.venv\Scripts\activate     # Windows

# Zainstaluj zależności
pip install -r requirements.txt
```

---

## Retrenowanie modeli

### WAŻNE: Konfiguracja ścieżek

**Przed uruchomieniem skryptów treningowych**, dostosuj ścieżki w każdym pliku `Final_v2_*.py`:

Na **samym końcu** każdego skryptu znajdziesz:
```python
# ==================== KONFIGURACJA ŚCIEŻEK ====================
MODELS_DIR = r'C:\Users\cinek\OneDrive\Pulpit\Studia\Inzynierka\LSTMS\models'
GRAPHS_DIR = r'C:\Users\cinek\OneDrive\Pulpit\Studia\Inzynierka\LSTMS\graphs'
# ==============================================================
```

**Zmień te ścieżki** na swoje lokalne katalogi:
```python
MODELS_DIR = r'C:\Twoja\Sciezka\Do\models'
GRAPHS_DIR = r'C:\Twoja\Sciezka\Do\graphs'
```

### Trenowanie
```bash
# Trenuj wybrany model (przykład dla 5d)
python Final_v2_5d.py
```


Każdy skrypt wytrenuje **7 modeli** z różnymi inicjalizacjami, wybierze **3 najlepsze** (ensemble) i zapisze je wraz z:
- Scalerem (`scaler.pkl`)
- Konfiguracją (`config.json`)
- Wykresami ewaluacji (w `graphs/`)

---

## Wykresy ewaluacji

Folder `graphs/` zawiera 3 typy wizualizacji dla każdego modelu:

1. **`predicted_vs_actual.png`** – Rzeczywiste vs przewidywane ceny (scatter plot)
2. **`prediction_errors.png`** – Rozkład błędów predykcji (histogram + time series)
3. **`cumulative_returns.png`** – Skumulowane zwroty (strategia vs faktyczne)

Wykresy są automatycznie generowane podczas treningu i zapisywane w odpowiednich podkatalogach.

---

## Korekcja systematycznego bias

### Problem

Modele wytrenowane na danych historycznych (2015-2023) wykazują systematyczne **niedoszacowanie** cen w środowisku produkcyjnym (2025), ponieważ cena AAPL wzrosła z ~$100 (mediana treningowa) do ~$230 (produkcja).

### Rozwiązanie

Plik `bias_corrections.json` zawiera współczynniki korekcji dla każdego modelu:
```json
{
  "1d": {
    "bias_correction_factor": 1.0142,
    "scaler_center": 171.23,
    "scaler_scale": 12.45
  },
  "5d": {
    "bias_correction_factor": 1.0089,
    ...
  }
}
```

**Zastosowanie w kodzie:**
```python
import json

# Załaduj współczynniki
with open('bias_corrections.json', 'r') as f:
    corrections = json.load(f)

# Zastosuj korekcję
raw_prediction = model.predict(X_input)
corrected = raw_prediction * corrections['5d']['bias_correction_factor']
```

Więcej szczegółów w skrypcie `bias_correction.py`.

---

## Różnice między modelami 1d i 5d

 **Modele 1d i 5d używają tej samej architektury** (Bidirectional LSTM-GRU). Różnice:

| Parametr | Model 1d | Model 5d |
|----------|----------|----------|
| **Sequence Length** | **40 dni** | **60 dni** |
| **Ścieżka do modeli** | `models/1d/` | `models/5d/` |
| **Architektura** | Identyczna | Identyczna |
| **Hiperparametry** | Identyczne | Identyczne |



---

##  Model Ridge Regression

Plik `new_RR_1.py` zawiera implementację modelu **Ridge Regression** do porównania z LSTM-GRU. Model jest trenowany dynamicznie (on-demand) i służy jako baseline.

**Użycie:**
```bash
python new_RR_1.py
```

**Wyniki:** Model Ridge osiąga dobre rezultaty tylko dla horyzontu **1-3 dni** (MAE ~$3, R² ~0.94), następnie jakość drastycznie spada ze względu na rekursywną metodę predykcji.

---

## Dokumentacja naukowa

Szczegółowy opis architektury, metodologii treningu oraz wyników eksperymentów znajduje się w pracy dyplomowej.

---

## Ograniczenia i uwagi

1. **Tylko AAPL:** Modele były trenowane wyłącznie na akcjach Apple Inc. Generalizacja na inne spółki nie była testowana.

2. **Brak danych fundamentalnych:** Modele bazują wyłącznie na analizie technicznej (OHLCV + wskaźniki techniczne). Nie uwzględniają raportów kwartalnych, newsów, sentymentu.

3. **Data wytrenowania:** Modele zostały wytrenowane na danych do **18 listopada 2025**. Dla danych produkcyjnych po tej dacie zaleca się retrenowanie.

4. **Horyzont 21 dni:** Model długoterminowy ma ograniczoną użyteczność (R² < 0). Dla tak długich prognoz zaleca się włączenie danych fundamentalnych.

5. **Korekcja bias:** W środowisku produkcyjnym **konieczne** jest zastosowanie współczynników korekcji z `bias_corrections.json`.

---

## 🔗 Powiązane repozytoria

- 🌐 **[Web Application (Frontend + Backend)]([https://github.com/twoj-username/stock-prediction-webapp](https://github.com/MBCstudio/Praca_Inzynierska_UI))** – Aplikacja webowa React + Flask wykorzystująca te modele

---

## Kontakt

**Autor:** Marcin Borkowski  
 

---

## Licencja

Projekt udostępniony na licencji **MIT License**. Zobacz [LICENSE](LICENSE).

---
