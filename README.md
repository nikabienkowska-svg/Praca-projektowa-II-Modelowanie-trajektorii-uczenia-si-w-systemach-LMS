# 🎓 Modelowanie Trajektorii Uczenia się w Systemach LMS (OULAD)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Projekt z przedmiotu Praca Projektowa II – Specjalizacja Sztuczna Inteligencja**  
> **Kategoria tematyczna:** 4. Rekurencyjne Sieci Neuronowe (RNN, LSTM, GRU) — *Maksymalna ocena: 5.0*

---

## 📌 1. Opis i Cel Projektu

W edukacji zdalnej i systemach e-learningowych (LMS, np. Moodle, Canvas, Open University) kluczowym wyzwaniem jest zjawisko **wczesnego porzucania kursów przez studentów (dropout)**. 

W odróżnieniu od klasycznych modeli tabelarycznych (które analizują jedynie statyczne cechy zagregowane), w niniejszym projekcie modelujemy **pełną dynamikę temporalną zachowań** studentów w czasie. Wykorzystujemy rekurencyjne sieci neuronowe (**LSTM**, **GRU**) do modelowania sekwencji interakcji w środowisku LMS:
$$\text{Oglądanie materiału} \longrightarrow \text{Przerwa czasowa} \longrightarrow \text{Aktywność na forum} \longrightarrow \text{Quiz / Ocena}$$

### Główne cele badawcze:
1. **Predykcja ryzyka porzucenia kursu (*Early Dropout Detection*)**: Wczesne przewidywanie, którzy użytkownicy dokonają wyrejestrowania / porzucenia kursu w najbliższych **7 dniach**.
2. **Identyfikacja wzorca kryzysu motywacyjnego**: Wykrycie anomalii i charakterystycznych zmian w zachowaniu poprzedzających rezygnację (np. nagłe wydłużenie przerw między logowaniami, zmiana proporcji aktywności biernej do czynnej, załamanie wyników testów).
3. **Benchmarking**: Porównanie wyników modeli sekwencyjnych (LSTM, BiLSTM, GRU) z silnymi modelami bazowymi (Regresja Logistyczna, Random Forest, XGBoost).

---

## 📊 2. Zbiór Danych (OULAD)

Projekt bazuje na renomowanym zbiorze **Open University Learning Analytics Dataset (OULAD)**:
* 🔗 **Strona projektu**: [Open University Learning Analytics Dataset](https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/)
* 🔗 **Zenodo**: [https://zenodo.org/records/14003233](https://zenodo.org/records/14003233)
* 🔗 **Kaggle**: [OULAD on Kaggle](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)

Zbiór zawiera dane o ponad 32 000 studentów, ich interakcjach z wirtualnym środowiskiem nauki (ponad 10 mln kliknięć) oraz wynikach zaliczeń. Dokładny schemat tabel i instrukcję pobrania znajdziesz w pliku [`data/README.md`](data/README.md).

---

## 👥 3. Podział Pracy w Zespole

Poniższa struktura pozwala na równoległą pracę każdego członka zespołu bez wchodzenia sobie w drogę.

### Role funkcjonalne:

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│     1. DATA & PIPELINE LEAD     │      │   2. DEEP LEARNING & MODELING   │      │ 3. BASELINE, XAI & EVALUATION   │
├─────────────────────────────────┤      ├─────────────────────────────────┤      ├─────────────────────────────────┤
│ • Pobranie i czyszczenie OULAD  │ ───► │ • Architektura LSTM i GRU       │ ───► │ • Modele bazowe (RF, XGBoost)   │
│ • Konstrukcja sekwencji (N,T,F) │      │ • PyTorch DataLoader & Padded   │      │ • Metryki: ROC-AUC, PR-AUC, F1  │
│ • Definicja okna targetu (7 dni)│      │ • Trening, Early Stopping, Loss │      │ • Analiza kryzysu motywacji     │
└─────────────────────────────────┘      └─────────────────────────────────┘      └─────────────────────────────────┘
```

| Rola w zespole | Odpowiedzialność | Główne pliki i moduły |
| :--- | :--- | :--- |
| **Inżynier Danych (Data & Feature Pipeline)** | Przygotowanie danych, łączenie tabel, agregacja interakcji temporalnych, budowa tensorów sekwencyjnych $(N, T, F)$, definicja 7-dniowego targetu. | `src/data/`, `src/features/`, `notebooks/01_eda_...`, `notebooks/02_eda_...` |
| **Specjalista Deep Learning (Modeling Lead)** | Implementacja modeli rekurencyjnych (LSTM, GRU, BiLSTM) w PyTorch, obsługa zmiennych długości sekwencji (`pack_padded_sequence`), pętla treningowa. | `src/models/lstm_model.py`, `src/models/train.py`, `notebooks/04_lstm_...` |
| **Analityk Ewaluacji i XAI (Evaluation & Insights)** | Klasyczne modele bazowe (Baseline), strojenie progów decyzyjnych przy niezbalansowanym zbiorze, analiza załamania motywacji, wykresy i raport. | `src/models/baselines.py`, `src/utils/metrics.py`, `src/interpretability/`, `reports/` |

---

## 🗂️ 4. Struktura Projektu

```text
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── task.md                 # Szablon nowego zadania
│   │   └── bug_report.md           # Szablon zgłoszenia błędu
│   └── pull_request_template.md    # Szablon zgłoszenia zmian (PR)
├── config/
│   └── config.yaml                 # Centralna konfiguracja parametrów i ścieżek
├── data/
│   ├── raw/                        # Pobrane surowe pliki CSV (ignorowane w Git)
│   ├── processed/                  # Wygenerowane tensory i próbki (ignorowane w Git)
│   └── README.md                   # Opis tabel OULAD i źródeł pobrania
├── notebooks/
│   ├── 01_eda_student_demographics_and_courses.ipynb
│   ├── 02_eda_temporal_vle_interactions.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_lstm_gru_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── data/                       # Wczytywanie i walidacja danych
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── features/                   # Agregacja czasowa i generowanie sekwencji
│   │   ├── __init__.py
│   │   └── sequence_builder.py
│   ├── models/                     # Definicje modeli ML i PyTorch RNN
│   │   ├── __init__.py
│   │   ├── lstm_model.py
│   │   └── train.py
│   ├── interpretability/           # Analiza wzorca kryzysu motywacyjnego
│   │   ├── __init__.py
│   │   └── motivation_patterns.py
│   └── utils/                      # Metryki ewaluacyjne (ROC-AUC, PR-AUC itp.)
│       ├── __init__.py
│       └── metrics.py
├── reports/
│   └── figures/                    # Wygenerowane wykresy do raportu i prezentacji
├── .gitignore                      # Zabezpieczenie przed wrzucaniem danych i wag
├── requirements.txt                # Lista zależności Pythona
└── README.md                       # Główna dokumentacja projektu
```

---

## 🚀 5. Jak Zacząć (Szybki Start)

### 1. Klonowanie repozytorium:
```bash
git clone https://github.com/nikabienkowska-svg/Praca-projektowa-II-Modelowanie-trajektorii-uczenia-si-w-systemach-LMS.git
cd Praca-projektowa-II-Modelowanie-trajektorii-uczenia-si-w-systemach-LMS
```

### 2. Utworzenie i aktywacja środowiska wirtualnego:
```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# lub: .venv\Scripts\activate  # Windows
```

### 3. Instalacja zależności:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Pobranie danych:
Pobierz pliki `.csv` ze zbioru OULAD i umieść je w folderze `data/raw/` zgodnie ze wskazówkami w [`data/README.md`](data/README.md).

---

## 🤝 6. Dobre Praktyki Zespołowe na GitHubie

Aby praca zespołowa przebiegała bezkonfliktowo i w pełni profesjonalnie:

1. **Praca na gałęziach (Branching)**:
   * Nigdy nie commitujemy bezpośrednio do `main`.
   * Tworzymy gałęzie z prefiksami, np.:
     * `feature/data-loader`
     * `feature/lstm-architecture`
     * `analysis/eda-demographics`
     * `fix/padding-bug`
2. **Pull Requests (PR) & Code Review**:
   * Każda zmiana trafia do `main` przez PR z wykorzystaniem szablonu `.github/pull_request_template.md`.
   * Przynajmniej jedna inna osoba z zespołu musi sprawdzić i zatwierdzić kod przed mergem.
3. **Higiena Jupyter Notebooków**:
   * Kluczowy kod wielokrotnego użytku piszemy w modułach w `src/`.
   * W notebookach jedynie importujemy moduły i prezentujemy wykresy.
   * Przed commitem notebooka czyścimy wyjścia (`Clear All Outputs`) lub upewniamy się, że nie generuje konfliktów.
4. **Zarządzanie zadaniami (GitHub Projects & Issues)**:
   * Wszystkie zadania tworzymy jako **Issues** w zakładce projektu na GitHubie.
   * Korzystamy z tablicy **GitHub Projects (Kanban)** ze statusami: `Todo` ➔ `In Progress` ➔ `In Review` ➔ `Done`.
