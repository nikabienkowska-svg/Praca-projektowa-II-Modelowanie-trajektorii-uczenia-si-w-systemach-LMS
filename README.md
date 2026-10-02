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

## 👥 4. Podział Pracy w Zespole


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
| **Specjalista Modelowania Transformerów** | Fine-tuning modelu BERT do klasyfikacji perswazji, dobór hiperparametrów i trening na GPU (RTX 2080 Ti). | `src/model.py`, `notebooks/02_modelowanie_bert.ipynb` |
| **Analityk Ewaluacji i Sprawozdań** | Ewaluacja metryk (ROC-AUC, F1), wizualizacja ewolucji pewności w dyskusji, koordynacja raportów etapowych. | `RAPORTY/`, `notebooks/01_eksploracja_danych.ipynb` |

---

## 🗂️ 5. Struktura Projektu

Układ katalogów został zorganizowany według wymogów projektowych:

```text
├── RAPORTY/                       # 8 etapowych raportów projektowych (wg szablonu semestralnego)
│   ├── Raport_1_problem_i_hipotezy.md
│   ├── Raport_2_pozyskanie_danych.md
│   ├── Raport_3_eksploracja_i_czyszczenie.md
│   ├── Raport_4_jakosc_redukcja_wymiarow.md
│   ├── Raport_5_przeglad_podejsc.md
│   ├── Raport_6_implementacja.md
│   ├── Raport_7_wyniki_i_analiza_bledow.md
│   └── Raport_8_weryfikacja_hipotez_i_wnioski.md
├── data/
│   ├── raw/                       # Pobrane surowe pliki Webis-CMV (pairs.jsonl.bz2, author_liwc.jsonl.bz2)
│   ├── processed/                 # Gotowa próbka cmv_persuasion_pairs_sample.jsonl
│   └── README.md                  # Opis tabel i linki do Zenodo
├── notebooks/
│   ├── 01_eksploracja_danych.ipynb    # Wczytanie danych, statystyki słów, pierwsze wykresy
│   └── 02_modelowanie_bert.ipynb      # Trening modelu BERT
├── src/
│   ├── __init__.py
│   ├── loader.py                  # Prosta funkcja load_cmv_dataframe()
│   ├── features.py                # Zliczanie słów wahania (hedging) i pewności
│   ├── model.py                   # Klasa modelu BERT
│   └── prepare_cmv.py             # Skrypt rozpakowujący dane
├── PLAN_DZIALANIA.md              # Przejrzysta mapa drogowa projektu i terminy etapów
├── requirements.txt               # Zależności bibliotek w Pythonie
├── .gitignore                     # Blokada commitowania ciężkich baz i wag
└── README.md                      # Główny opis projektu
```

---

## 🚀 6. Jak Zacząć

1. Przeczytaj plik [`PLAN_DZIALANIA.md`](PLAN_DZIALANIA.md) — znajdziesz tam terminy i podział zadań na cały semestr.
2. Zapoznaj się z gotowym [`RAPORTY/Raport_1_problem_i_hipotezy.md`](RAPORTY/Raport_1_problem_i_hipotezy.md) oraz [`RAPORTY/Raport_2_pozyskanie_danych.md`](RAPORTY/Raport_2_pozyskanie_danych.md).
3. Dane są już pobrane i czekają w `data/processed/cmv_persuasion_pairs_sample.jsonl` — możesz od razu otworzyć `notebooks/01_eksploracja_danych.ipynb`!
