# OULAD - Open University Learning Analytics Dataset

Katalog `data/` służy do przechowywania danych źródłowych i przetworzonych.
> ⚠️ **Uwaga:** Pliki z danymi (`*.csv`, `*.parquet`, archiwa) są ignorowane przez `.gitignore` i **nie powinny być commitowane do repozytorium**.

---

## Źródło danych
Zbiór **OULAD (Open University Learning Analytics Dataset)** zawiera zanonimizowane dane z kursów prowadzonych na Open University:
* Oficjalna strona: [https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/](https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/)
* Zenodo: [https://zenodo.org/records/14003233](https://zenodo.org/records/14003233)
* Kaggle: [https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)

## Schemat i zawartość tabel

Pliki należy rozpakować do katalogu `data/raw/`:

1. `courses.csv` - informacje o modułach (kod modułu, semestr, długość kursu w dniach).
2. `assessments.csv` - oceny/zadania w ramach modułów (TMA - ocena tutora, CMA - test komputerowy, Exam - egzamin, waga, data oddania).
3. `vle.csv` - zasoby w Wirtualnym Środowisku Nauki (LMS) - materiały dydaktyczne, fora dyskusyjne, quizy, podstrony, linki.
4. `studentInfo.csv` - dane demograficzne studentów (płeć, wykształcenie, wiek, liczba wcześniejszych podejść, status końcowy: *Withdrawn*, *Fail*, *Pass*, *Distinction*).
5. `studentRegistration.csv` - daty zapisu na kurs i wyrejestrowania (`date_unregistration` - kluczowa dla targetu *dropout*).
6. `studentAssessment.csv` - wyniki uzyskane przez studentów z poszczególnych zadań i daty przesłania.
7. `studentVle.csv` - **najważniejsza tabela dla sekwencji temporalnych** - dzienne interakcje studenta z zasobami LMS (`id_site`, `date`, `sum_click`).

---

## Przetwarzanie (`data/processed/`)
W katalogu `data/processed/` generowane są przetworzone tensory i ramki danych:
* `student_daily_features.parquet` - dzienne zagregowane wektory aktywności studentów.
* `sequences_train.pt`, `sequences_val.pt`, `sequences_test.pt` - przygotowane sekwencje pod PyTorch DataLoader.
