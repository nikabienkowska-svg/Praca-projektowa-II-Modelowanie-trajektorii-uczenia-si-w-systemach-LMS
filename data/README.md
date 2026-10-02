# Webis-CMV-20 (ChangeMyView Corpus 2020)

Katalog `data/` zawiera dane do zadania predykcji perswazji, polaryzacji postawy oraz śledzenia ewolucji przekonań dyskutantów.

---

## 🔗 Źródło danych
* Oficjalna strona Webis: [https://webis.de/data/webis-cmv-20.html](https://webis.de/data/webis-cmv-20.html)
* Repozytorium Zenodo: [https://zenodo.org/records/3778298](https://zenodo.org/records/3778298) (DOI: `10.5281/zenodo.3778298`)

## 📁 Pobrane pliki surowe (`data/raw/`)
Pliki są w formacie skompresowanym `*.jsonl.bz2`:
1. `pairs.jsonl.bz2` (ok. 17.2 MB) — **10 303 sparowane przypadki argumentacji**:
   - `submission`: post inicjujący (tytuł, treść, autor).
   - `delta_comment`: kontrargument, który przekonał autora i otrzymał symbol Delty ($\Delta$).
   - `nodelta_comment`: kontrargument o podobnej tematyce, który NIE przekonał autora.
   - `comments_similarity`: miara podobieństwa leksykalnego między komentarzami w parze.
2. `author_liwc.jsonl.bz2` (ok. 14.0 MB) — cechy lingwistyczne LIWC autorów:
   - Wskaźniki pewności siebie (`certain`) vs zwrotów asekuracyjnych (`tentat`).
   - Ton emocjonalny (`tone`), afekt (`affect`), procesy poznawcze (`cogproc`).

## ⚙️ Przetworzone zbiory (`data/processed/`)
* `cmv_persuasion_pairs_sample.jsonl` — oczyszczona próbka par (gotowa do wczytania przez Pandas / HuggingFace `datasets`).
  Wygenerujesz ją poleceniem:
  ```bash
  python src/data/prepare_cmv.py --max_records 1000
  ```
