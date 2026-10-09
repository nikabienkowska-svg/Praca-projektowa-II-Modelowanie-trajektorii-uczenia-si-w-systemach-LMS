# Modelowanie trajektorii uczenia się w systemach LMS

Projekt polega na analizie danych z platformy edukacyjnej w celu wczesnego przewidywania, czy uczeń zrezygnuje z kursu. Wykorzystujemy do tego sieci neuronowe analizujące sekwencje aktywności (logi, przerwy, zadania).

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

## 📂 Struktura repozytorium

Przygotowaliśmy czystą przestrzeń do wypełniania w kolejnych tygodniach pracy:

- 📁 `RAPORTY/` - tutaj znajdują się oficjalne raporty (np. z pierwszych i drugich zajęć). Będziemy je tu dodawać tydzień po tygodniu.
- 📁 `data/` - miejsce na surowe zbiory danych OULAD, które dopiero tu pobierzemy.
- 📁 `notebooks/` - folder na notatniki Jupyter z eksploracją danych (zadania Kasi i Patrycji).
- 📁 `src/` - miejsce, gdzie będziemy wrzucać nasz kod (od wczytywania danych po same modele).

---

## 📖 Słownik pojęć i technologii

### Pojęcia
- **LMS (Learning Management System) / VLE** - platforma edukacyjna online, np. Moodle.
- **Dropout / Withdrawn (Porzucenie kursu)** - moment, w którym student wyrejestrowuje się z zajęć.
- **Kryzys motywacyjny** - etap zmniejszonego zaangażowania poprzedzający porzucenie kursu (mniej zadań, dłuższe przerwy).
- **OULAD (Open University Learning Analytics Dataset)** - publiczny zbiór danych używany do naszych analiz.

### Technologie
- **Python** - główny język programowania używany do analizy danych i modeli.
- **PyTorch** - biblioteka służąca do budowy i trenowania zaawansowanych sieci neuronowych.
- **LSTM (Long Short-Term Memory)** - architektura sieci neuronowych potrafiąca analizować sekwencje czasowe.
- **XGBoost / Random Forest** - "klasyczne" algorytmy uczenia maszynowego używane przez nas do stworzenia punktu odniesienia.
- **Pandas / Scikit-Learn** - biblioteki w Pythonie służące do zarządzania tabelami danych.
- **Venv (Virtual Environment)** - wyizolowane środowisko lokalne zapobiegające konfliktom z innymi aplikacjami w systemie.
