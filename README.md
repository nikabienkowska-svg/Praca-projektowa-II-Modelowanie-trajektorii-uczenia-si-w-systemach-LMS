# 💬 Predykcja Polaryzacji Postawy i Ewolucji Przekonań na Forach Dyskusyjnych (Webis-CMV-20 & BERT)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Projekt z przedmiotu Praca Projektowa II – Specjalizacja Sztuczna Inteligencja**  
> **Kategoria tematyczna:** 6. Transformery (Attention / BERT / GPT) — *Maksymalna ocena: 5.0*  
> **Koncentracja:** Dane tekstowe (język naturalny), sekwencje długoterminowe oraz kontekst semantyczny decyzji.

---

## 📌 1. Opis i Cel Projektu

W dyskusjach internetowych i mediach społecznościowych (szczególnie na forum **Reddit: r/ChangeMyView**) autor posta (**OP – Original Poster**) przedstawia swoje wyjściowe, silne przekonanie na dany temat. Inni użytkownicy formułują kontrargumenty w sekwencjach komentarzy. Gdy argument okaże się przekonujący, autor przyznaje komentatorowi symbol **Delty ($\Delta$)**, sygnalizując zmianę swojego stanowiska.

Celem projektu jest **zastosowanie i fine-tuning modeli Transformer (BERT / RoBERTa / DeBERTa)** do:
1. **Predykcji skuteczności perswazji**: Przewidywania, który z przedstawionych kontrargumentów doprowadzi do zmiany zdania i przyznania Delty ($\Delta$).
2. **Wykrywania ewolucji przekonań w sekwencji wypowiedzi**: Identyfikacji punktu krytycznego (*inflection point*), w którym język autora przechodzi ze stanu **kategorycznego / pewnego siebie** (*high certainty / low hedging*) w stan **wahania i zwątpienia** (*tentative / epistemic hedging*, np. *„perhaps”, „you have a point”, „I hadn't considered”*).
3. **Analizy mechanizmu atencji (Self-Attention)**: Wyjaśnienia, jakie cechy argumentacyjne i struktury pojęciowe najsilniej determinują polaryzację i zmianę postawy.

---

## 🔬 2. Pytania Badawcze i Hipotezy

### Pytania badawcze (Research Questions):
* **Q1 (Skuteczność perswazji):** Czy dwuwejściowy model BERT (`[CLS] Post OP [SEP] Kontrargument [SEP]`) pozwala na istotnie lepsze odróżnienie argumentów przekonujących ($\Delta$) od nieprzekonujących niż klasyczne modele lingwistyczne i TF-IDF?
* **Q2 (Dynamika emocji i pewności):** W jaki sposób zmienia się modalność epistemologiczna (wskaźniki *certainty* vs *tentative* z LIWC) w sekwencji kolejnych odpowiedzi autora przed momentem ustąpienia?
* **Q3 (Podobieństwo semantyczne a zmiana zdania):** Czy skuteczniejsza jest perswazja operująca na bliskich pojęciach (wysokie *cosine similarity* embeddingów z `sentence-transformers`), czy wprowadzająca zupełnie nową domenę pojęciową?
* **Q4 (Interpretowalność atencji):** Czy wagi mechanizmu Attention skupiają się na logicznych przesłankach i faktach, czy na nacechowanych emocjonalnie zwrotach erystycznych?

### Hipotezy badawcze (Hypotheses):
* **$H_1$ (Ewolucja modalności):** Zmianie zdania towarzyszy mierzalny wzrost wskaźnika zwrotów asekuracyjnych (*hedging density*) i jednoczesny spadek leksykalnych markerów pewności (*certainty markers*).
* **$H_2$ (Przewaga architektury Transformer):** Fine-tuned BERT osiągnie wynik ROC-AUC oraz F1 wyższy o co najmniej 10–15 p.p. w stosunku do klasycznych modeli bazowych (Regresja Logistyczna / Random Forest).
* **$H_3$ (Wymiar atencji w punkcie zwrotnym):** Tokeny o najwyższych wagach atencji w warstwach końcowych BERT-a odpowiadają bezpośrednio nowym informacjom i przesłankom przyczynowo-skutkowym.

---

## 📊 3. Zbiór Danych (Webis-CMV-20)

Projekt bazuje na korpusie **Webis-CMV-20 (ChangeMyView Corpus 2020)**:
* 🔗 **Strona Webis**: [https://webis.de/data/webis-cmv-20.html](https://webis.de/data/webis-cmv-20.html)
* 🔗 **Zenodo (DOI)**: [https://zenodo.org/records/3778298](https://zenodo.org/records/3778298)

### Przygotowane zbiory danych w projekcie:
* `data/raw/pairs.jsonl.bz2` — 10 303 sparowane przypadki (post OP, argument z Deltą $\Delta$, argument bez Delty).
* `data/raw/author_liwc.jsonl.bz2` — cechy psycholingwistyczne autorów (LIWC: *certain*, *tentat*, *tone*, *affect*).
* `data/processed/cmv_persuasion_pairs_sample.jsonl` — oczyszczona próbka 1 000 par gotowa do natychmiastowego modelowania.

---

## 🔄 4. Ponowne Wykorzystanie Narzędzi z Poprzedniego Projektu

Projekt bezpośrednio wykorzystuje sprawdzone komponenty z poprzedniej pracy projektowej ([`nikabienkowska-svg/Praca-projektowa-`](https://github.com/nikabienkowska-svg/Praca-projektowa-)):
1. **`sentence-transformers` & `transformers`**:
   - Pipeline do generowania embeddingów semantycznych (`all-MiniLM-L6-v2`) i wyliczania podobieństwa kosinusowego argumentów.
2. **`scikit-learn`**:
   - Sprawdzony pipeline ewaluacji (ROC-AUC, PR-AUC, F1, macierz pomyłek) dla modeli bazowych.
3. **`Gradio` (Aplikacja demonstracyjna)**:
   - Budowa interaktywnego panelu dla prowadzącego/recenzenta: wklejenie posta i kontrargumentu z wizualizacją prawdopodobieństwa perswazji i wskaźnika pewności w czasie rzeczywistym!

---

## 👥 5. Podział Pracy w Zespole

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│     1. DATA & LINGUISTICS       │      │   2. TRANSFORMERS & MODELING    │      │  3. INTERPRETABILITY & GRADIO   │
├─────────────────────────────────┤      ├─────────────────────────────────┤      ├─────────────────────────────────┤
│ • Przygotowanie Webis-CMV-20    │ ───► │ • Fine-tuning BERT / RoBERTa    │ ───► │ • Modele bazowe (RF, Logistic)  │
│ • Analiza markerów hedging/LIWC │      │ • HuggingFace Trainer / PyTorch │      │ • Analiza wag Attention i XAI   │
│ • Ekstrakcja sekwencji dyskusji │      │ • Ewaluacja ROC-AUC, F1-score   │      │ • Interaktywne demo w Gradio    │
└─────────────────────────────────┘      └─────────────────────────────────┘      └─────────────────────────────────┘
```

| Rola w zespole | Zakres odpowiedzialności | Pliki i moduły |
| :--- | :--- | :--- |
| **Inżynier Danych i Lingwistyki NLP** | Czyszczenie i przygotowanie par CMV, ekstrakcja markerów zwątpienia (*hedging*) i pewności (*certainty*), łączenie z cechami LIWC, analiza EDA. | `src/data/`, `src/features/text_features.py`, `notebooks/01_eda_...`, `notebooks/02_linguistic_...` |
| **Specjalista Modelowania Transformerów** | Fine-tuning modelu BERT / RoBERTa do klasyfikacji sekwencji par, optymalizacja hiperparametrów (learning rate, warm-up, weight decay), benchmarking. | `src/models/bert_classifier.py`, `notebooks/04_bert_fine_tuning_...` |
| **Analityk Interpretowalności i Wdrożenia (XAI & Demo)** | Modele bazowe (TF-IDF + Random Forest / Regresja Logistyczna), śledzenie trajektorii przekonań, wizualizacja wag atencji oraz interaktywne demo w **Gradio**. | `src/models/baselines.py`, `src/interpretability/belief_trajectory.py`, `notebooks/05_belief_evolution_...` |

---

## 🗂️ 6. Struktura Projektu

```text
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── task.md                 # Szablon nowego zadania zespołowego
│   │   └── bug_report.md           # Szablon zgłoszenia błędu
│   └── pull_request_template.md    # Szablon PR (wymóg checklisty przed scaleniem)
├── config/
│   └── config.yaml                 # Centralna konfiguracja (BERT, ścieżki, markery)
├── data/
│   ├── raw/                        # Pobrane surowe zbiory pairs.jsonl.bz2, author_liwc.jsonl.bz2
│   ├── processed/                  # Wygenerowana próbka cmv_persuasion_pairs_sample.jsonl
│   └── README.md                   # Opis zbioru Webis-CMV-20
├── notebooks/
│   ├── 01_eda_webis_cmv_overview.ipynb
│   ├── 02_linguistic_certainty_and_liwc_analysis.ipynb
│   ├── 03_baseline_models_tfidf_and_embeddings.ipynb
│   ├── 04_bert_fine_tuning_and_persuasion_prediction.ipynb
│   └── 05_belief_evolution_and_gradio_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # Wczytywanie przetworzonych rekordów CMV
│   │   └── prepare_cmv.py          # Skrypt przetwarzający surowe bz2 do czystego JSONL
│   ├── features/
│   │   ├── __init__.py
│   │   └── text_features.py        # Markery hedgingu, certainty i sentence-transformers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py            # Modele klasyczne (TF-IDF, Regresja, Random Forest)
│   │   ├── bert_classifier.py      # Fine-tuning BERT do klasyfikacji perswazji
│   │   └── train.py                # Skrypt treningowy baseline'ów
│   ├── interpretability/
│   │   ├── __init__.py
│   │   └── belief_trajectory.py    # Śledzenie ewolucji przekonań i punktu zwrotnego
│   └── utils/
│       ├── __init__.py
│       └── metrics.py              # Metryki: ROC-AUC, PR-AUC, F1, Confusion Matrix
├── tests/
│   └── test_pipeline.py            # Test jednostkowy spójności modułów NLP i modeli
├── .gitignore                      # Ochrona przed wrzucaniem dużych danych, wag i cache
├── requirements.txt                # Biblioteki (Transformers, Sentence-Transformers, PyTorch, Gradio)
└── README.md                       # Główna dokumentacja projektu
```

---

## 🚀 7. Jak Zacząć (Szybki Start)

### 1. Klonowanie repozytorium:
```bash
git clone https://github.com/nikabienkowska-svg/Praca-projektowa-II-Modelowanie-trajektorii-uczenia-si-w-systemach-LMS.git
cd Praca-projektowa-II-Modelowanie-trajektorii-uczenia-si-w-systemach-LMS
```

### 2. Utworzenie środowiska wirtualnego:
```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# lub: .venv\Scripts\activate  # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Przygotowanie danych testowych:
Zbiór par został już pobrany do `data/raw/pairs.jsonl.bz2`. Aby wygenerować próbkę do natychmiastowej pracy:
```bash
python src/data/prepare_cmv.py --max_records 1000
```
