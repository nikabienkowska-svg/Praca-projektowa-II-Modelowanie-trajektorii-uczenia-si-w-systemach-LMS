# Modelowanie trajektorii uczenia się w systemach LMS

Cześć! Tutaj znajduje się nasz projekt grupowy. W skrócie: tworzymy program, który będzie przewidywał, czy uczeń rzuci studia/kurs, na podstawie tego jak często klika w platformie edukacyjnej.

---

## 👥 Zespół i podział ról

1. **Patrycja Goźlińska (Sekretarz)**
   - Prowadzenie dokumentacji projektu.
   - Tworzenie i organizowanie cotygodniowych raportów.

2. **Katarzyna Rogozińska (Badacz)**
   - Eksploracja danych i szukanie trendów (tzw. wzorców "kryzysu motywacyjnego").
   - Analizowanie wyników działania algorytmów.

3. **Weronika Bieńkowska (Moderator)**
   - Programowanie i testowanie modeli sztucznej inteligencji (LSTM, XGBoost).
   - Zarządzanie strukturą i organizacją całego repozytorium na GitHubie.

---

## 📂 Co tu w ogóle jest? (Przewodnik po folderach)

Uprościliśmy wszystko maksymalnie, żeby nikt się nie zgubił:

- 📁 `RAPORTY/` - **Wysyłamy to prowadzącemu.** Tutaj wrzucamy oficjalne pliki Word (`.docx`), które oddajemy co tydzień na zajęciach. 
- 📁 `data/` - **Nasz wielki magazyn danych.** Tutaj lądują potężne pliki Excel/CSV, z których będzie uczył się nasz model. Ze względu na swój rozmiar, te pliki żyją **tylko na naszych komputerach** i nie wysyłają się na GitHuba.
- 📁 `notebooks/` - **Brudnopis badacza.** Tu Kasia i Patrycja będą otwierać notatniki (pliki `.ipynb`), rysować wykresy i patrzeć "co w tych danych piszczy".
- 📁 `src/` - **Serce projektu (kod).** Tu Weronika trzyma skrypty. Obecnie jest tu tylko plik `pobierz_dane.py` - sprytny programik, który po odpaleniu sam pobiera dziewczynom pliki do folderu `data/`.
- 📄 `README.md` - to jest ten plik, który właśnie czytasz!

---

## 📖 Słownik pojęć i technologii (Jak tego używamy?)

### Z czym to się je na uczelni?
- **OULAD** - wielka, darmowa baza danych o brytyjskich studentach. To nasze jedyne źródło wiedzy o tym, kto i kiedy klikał w Moodle'u.
- **LMS (np. Moodle) / VLE** - po prostu uczelniana platforma. Nas interesuje, kiedy uczeń tam wszedł, a kiedy zniknął.
- **Dropout** - moment, kiedy student wypisał się z kursu na zawsze. Próbujemy przewidzieć ten moment 7 dni wcześniej.

### Magia informatyczna (Programowanie)
- **Python** - język w którym wszystko to piszemy. 
- **PyTorch** - gotowe "klocki Lego", z których budujemy naszą sieć neuronową.
- **LSTM** - mądra sieć neuronowa. Zapamiętuje nie tylko "ile" student wyklikał, ale też "w jakiej kolejności".
- **Pandas** - taki bardzo zaawansowany, ukryty pod kodem Pythona Excel do filtrowania i łączenia naszych danych.
