# Raport 2 — Pozyskanie danych
**Etap:** zajęcia 3–4 · **Temat 6:** Transformery — Predykcja polaryzacji postawy (Webis-CMV-20 / BERT)
**Zespół:** Weronika, [Członek zespołu 2], [Członek zespołu 3]

---

## 1. Cel etapu

Pozyskanie i opisanie zbioru danych: źródło, sposób pozyskania, liczba obserwacji, struktura rekordu oraz potencjalne problemy jakościowe. Efektem ma być gotowy, udokumentowany zbiór do dalszej analizy.

## 2. Wykonane działania

- **Wybór źródła:** Publiczny korpus naukowy **Webis-CMV-20** udostępniony przez grupę badawczą Webis (Bauhaus-Universität Weimar, University of Groningen, Leipzig University) w serwisie Zenodo pod adresem: `https://doi.org/10.5281/zenodo.3778298`.
- **Pobranie danych:**
  - `pairs.jsonl.bz2` (17.2 MB, skompresowany bzip2) — główny zbiór do zadania predykcji perswazyjności.
  - `author_liwc.jsonl.bz2` (14.0 MB, skompresowany bzip2) — zestaw cech psycholingwistycznych autorów.
- **Konwersja i filtracja:** Napisanie skryptu `src/prepare_cmv.py` parsującego surowy format `.jsonl.bz2` do czystej, zbalansowanej próbki roboczej `data/processed/cmv_persuasion_pairs_sample.jsonl`.
- **Kontrola spójności danych:** Sprawdzenie obecności kluczowych pól (`title`, `op_text`, `delta_argument`, `nodelta_argument`).

## 3. Uzyskane rezultaty

### Charakterystyka korpusu

| Cecha | Wartość |
|---|---|
| Łączna liczba par w `pairs.jsonl.bz2` | **10 303** par dyskusji |
| Liczba rekordów w próbce roboczej | **1 000** par |
| Format źródłowy | JSON Lines skompresowany bzip2 |
| Język wypowiedzi | angielski |
| Równowaga klas | 50% argumentów skutecznych ($\Delta$), 50% nieskutecznych |

### Struktura rekordu roboczego (`cmv_persuasion_pairs_sample.jsonl`):
- `submission_id`: identyfikator wątku na platformie Reddit.
- `author`: pseudonim autora posta (OP).
- `title`: tytuł wątku definiujący wyjściową tezę autora.
- `op_text`: pełny tekst uzasadnienia tezy przez OP.
- `delta_argument`: treść kontrargumentu, który doprowadził do przyznania nagrody $\Delta$.
- `nodelta_argument`: treść kontrargumentu o zbliżonej tematyce, który NIE przekonał autora.
- `similarity_score`: miara podobieństwa leksykalnego między parowanymi komentarzami.

## 4. Problemy jakościowe i ograniczenia

- **Ekstremalne długości postów:** Niektóre posty inicjujące zawierają rozbudowane eseje (nawet ponad 20 000 znaków), co wymaga odpowiedniej strategii obcinania (*truncation*) przy tokenizacji dla modelu BERT (limit 512 tokenów).
- **Formatowanie Markdown i cytaty:** Komentarze na Reddit obfitują w znaczniki cytatów (`> cytat`), linki URL i formatowanie, które wymagają standaryzacji przed podaniem do modelu.

## 5. Plan działań na kolejny etap (Raport 3)

1. Przeprowadzenie eksploracyjnej analizy danych (EDA) w notebooku.
2. Analiza rozkładów długości tekstów i słownictwa.
3. Czyszczenie tekstu (usunięcie zbędnych znaków formatowania Reddit, standaryzacja linków).
