
# Modelowanie trajektorii uczenia się w systemach LMS

Celem badawczym i praktycznym naszej pracy jest zbudowanie systemu wczesnego ostrzegania (Early Warning System), który na podstawie śladów cyfrowych studenta w platformie e-learningowej (np. Moodle) przewidzi ryzyko porzucenia kursu z N-dniowym wyprzedzeniem. Skupiamy się na analizie sekwencji interakcji w czasie (tzw. trajektorii uczenia się), aby zidentyfikować wzorce zachowań poprzedzające spadek motywacji.

Projekt znajduje się obecnie w fazie rozwoju (Work In Progress).

---


## 📂 Struktura repozytorium

* 📁 `RAPORTY/` - Miejsce docelowe dla oficjalnej dokumentacji. Zawiera zatwierdzone pliki tekstowe oddawane cyklicznie w ramach zajęć.
* 📁 `data/` - Przestrzeń na surowe i przetworzone dane. Ze względu na duży wolumen danych (zbiór główny posiada ponad 10 milionów rekordów), zawartość tego folderu nie jest śledzona przez system kontroli wersji (dodana do `.gitignore`).


* 📁 `notebooks/` - Przestrzeń eksperymentalna (pliki `.ipynb`). Służy do eksploracyjnej analizy danych (EDA), wizualizacji rozkładów zmiennych oraz wstępnego testowania logiki.
* 📁 `src/` - Kod źródłowy projektu. Obecnie znajduje się tu moduł `pobierz_dane.py` automatyzujący pobieranie niezbędnych plików do folderu `data/`. W przyszłości trafią tu moduły odpowiedzialne za inżynierię cech i trenowanie modeli.
* 📄 `README.md` - Plik wprowadzający, który właśnie czytasz.

---

## 📊 Dane i technologie

### Zbiór danych

Projekt opiera się na bazie **Open University Learning Analytics Dataset (OULAD)**.

* Są to dane zebrane z wirtualnego środowiska nauczania (VLE) brytyjskiej uczelni w latach 2013-2014.


* Rejestrują one szczegółowe logi interakcji studentów z różnymi typami zasobów edukacyjnych (np. rozwiązywanie quizów, przeglądanie forów, czytanie materiałów).


* Naszą zmienną docelową (target) jest moment formalnego wyrejestrowania się studenta z kursu (tzw. dropout).



### Wykorzystywane narzędzia

* **Python:** Główny język programowania wykorzystywany w analizie i modelowaniu.
* **Pandas:** Biblioteka do zaawansowanego przetwarzania, filtrowania i agregacji danych tabelarycznych w szeregi czasowe.


* **PyTorch:** Framework do budowy modeli głębokiego uczenia. Użyjemy go do zaimplementowania sieci **LSTM (Long Short-Term Memory)**, która potrafi zapamiętywać sekwencje i chronologię zdarzeń ucznia.


* **XGBoost / Random Forest:** Klasyczne modele uczenia maszynowego, operujące na cechach zagregowanych, które posłużą nam jako modele bazowe (baseline) do porównania skuteczności z siecią LSTM.



---

## 🚀 Uruchomienie projektu (Instrukcja w przygotowaniu - WIP)

*Poniższa sekcja będzie aktualizowana wraz z rozwojem skryptów w folderze `src/`.*

Na ten moment, aby przygotować środowisko pracy lokalnej, wykonaj następujące kroki:

1. **Sklonuj repozytorium:**
```bash
git clone https://github.com/nikabienkowska-svg/Praca-projektowa-II-Modelowanie-trajektorii-uczenia-si-w-systemach-LMS.git
cd Praca-projektowa-II-Modelowanie-trajektorii-uczenia-si-w-systemach-LMS

```


2. **Pobierz dane:**
Uruchom przygotowany skrypt, który pobierze bazę danych OULAD bezpośrednio do odpowiedniego folderu, z którego będą korzystać nasze notatniki.
```bash
python src/pobierz_dane.py

```


3. Otwórz pliki `.ipynb` w folderze `notebooks/`, by zapoznać się z pierwszymi wynikami eksploracji.
