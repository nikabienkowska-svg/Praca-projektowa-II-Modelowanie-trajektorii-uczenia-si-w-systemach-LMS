# 🗺️ Plan Działania — Praca Projektowa II
**Temat 6:** Transformery (Attention / BERT / GPT) — Predykcja polaryzacji postawy (Webis-CMV-20 / BERT)  
**Skład zespołu:** 3 osoby

Ten dokument to nasza prosta mapa drogowa na cały semestr. Każdy etap odpowiada jednemu raportowi cząstkowemu (dokładnie tak, jak w poprzednim semestrze).

---

## 📌 Przegląd etapów i terminów raportów

| Etap / Raport | Termin (zajęcia) | Co robimy (po ludzku)? | Główny plik / folder | Status |
|:---|:---:|:---|:---|:---:|
| **[Raport 1](RAPORTY/Raport_1_problem_i_hipotezy.md)** | 1–2 | Wybór tematu, cel, hipotezy badawcze (H1, H2, H3) | `RAPORTY/Raport_1_...` | ✅ **Gotowy** |
| **[Raport 2](RAPORTY/Raport_2_pozyskanie_danych.md)** | 3–4 | Pobranie i opis danych Webis-CMV-20 (10 303 par) | `data/`, `RAPORTY/Raport_2_...` | ✅ **Gotowy** |
| **[Raport 3](RAPORTY/Raport_3_eksploracja_i_czyszczenie.md)** | 5–6 | Eksploracja danych (EDA), wykresy słów, czyszczenie tekstu | `notebooks/01_eksploracja_danych.ipynb` | ⏳ Następny |
| **[Raport 4](RAPORTY/Raport_4_jakosc_redukcja_wymiarow.md)** | 7–8 | Wyciągnięcie cech lingwistycznych (wahania, pewność, LIWC) | `src/features.py` | 📝 Do zrobienia |
| **[Raport 5](RAPORTY/Raport_5_przeglad_podejsc.md)** | 9–10 | Wybór modeli: prosty Baseline vs architektura BERT | `RAPORTY/Raport_5_...` | 📝 Do zrobienia |
| **[Raport 6](RAPORTY/Raport_6_implementacja.md)** | 11–12 | Trening BERT-a na stacjonarce (RTX 2080 Ti) + Baseline | `src/model.py`, `notebooks/02_...` | 📝 Do zrobienia |
| **[Raport 7](RAPORTY/Raport_7_wyniki_i_analiza_bledow.md)** | 13–14 | Wyniki (ROC-AUC, F1), macierz pomyłek, analiza błędów | `RAPORTY/Raport_7_...` | 📝 Do zrobienia |
| **[Raport 8](RAPORTY/Raport_8_weryfikacja_hipotez_i_wnioski.md)** | Zaliczenie | Sprawdzenie czy hipotezy się potwierdziły, wnioski końcowe | `RAPORTY/Raport_8_...` | 📝 Do zrobienia |

---

## 👥 Podział ról w zespole (Model PBL)

Aby praca szła sprawnie i nikt nie musiał robić wszystkiego sam:
1. **Osoba 1 (Inżynier Danych / EDA):**
   - Eksploracja danych w notebooku, wykresy długości postów i słownictwa (Etap 3).
   - Współtworzenie Raportu 3.
2. **Osoba 2 (Modelowanie i Cechy / Baseline):**
   - Przygotowanie cech lingwistycznych i prostego modelu bazowego (Etap 4 i 5).
   - Współtworzenie Raportu 4 i 5.
3. **Osoba 3 (Głębokie Uczenie / BERT & RTX 2080 Ti):**
   - Przygotowanie skryptu treningowego BERT-a i odpalenie go na GPU (Etap 6 i 7).
   - Współtworzenie Raportu 6 i 7.
*Raport 8 i sprawozdanie końcowe piszemy wspólnie na podstawie uzyskanych wyników.*

---

## 🚀 Jak zacząć pracę z danymi (dla każdej z nas)

1. Sklonuj repozytorium na swój komputer:
   ```bash
   git clone https://github.com/nikabienkowska-svg/Praca-projektowa-II-Modelowanie-trajektorii-uczenia-si-w-systemach-LMS.git
   cd Praca-projektowa-II-Modelowanie-trajektorii-uczenia-si-w-systemach-LMS
   ```
2. Dane są już pobrane w folderze `data/processed/cmv_persuasion_pairs_sample.jsonl`.
3. Otwórz notebook `notebooks/01_eksploracja_danych.ipynb` w Jupyterze, VS Code lub PyCharmie i kliknij „Uruchom”.
