# RAPORT NA ZAJĘCIA 1–2 (ETAP 1)
## Zdefiniowanie problemu i hipotez badawczych

| Metadana | Opis |
| :--- | :--- |
| **Temat projektu** | Modelowanie trajektorii uczenia się w systemach LMS (np. Moodle). Sieci LSTM analizujące sekwencje interakcji (oglądanie materiału → przerwa → quiz → wynik) w celu przewidzenia, którzy studenci dokonają „porzucenia kursu” w ciągu najbliższych 7 dni, oraz identyfikacja wzorca poprzedzającego kryzys motywacyjny. |
| **Zbiór danych** | Open University Learning Analytics Dataset (OULAD) – wersja z platformy Kaggle / UCI Machine Learning Repository (Kuzilek et al., Nature Scientific Data 2017). |
| **Zespół i role** | Patrycja Goźlińska (Sekretarz), Katarzyna Rogozińska (Badacz), Weronika Bieńkowska (Moderator) |

---

## 1. Cel etapu i problematyka badawcza

### 1.1. Tło problemu i uzasadnienie dziedzinowe
Rozwój platform e-learningowych oraz systemów zarządzania nauką LMS (*Learning Management System*, takich jak Moodle czy Canvas) sprawił, że edukacja zdalna i hybrydowa stała się powszechnym standardem akademickim. Jednym z najpoważniejszych wyzwań dydaktycznych i organizacyjnych pozostaje jednak wysoki odsetek przedwczesnych rezygnacji z kursów (*student dropout / withdrawal*).

W zbiorze OULAD (*Open University Learning Analytics Dataset*), obejmującym 32 593 rejestracje studenckie (28 785 unikalnych studentów) w 22 edycjach kursów, aż **31,16% rejestracji (10 156 przypadków)** kończy się formalnym wyrejestrowaniem przed zakończeniem modułu (status `Withdrawn`).

Z perspektywy psychologii edukacji porzucenie kursu rzadko bywa decyzją nagłą. Jest ono niemal zawsze punktem kulminacyjnym stopniowego procesu wycofywania się, który roboczo definiujemy jako **kryzys motywacyjny**. Choć motywacja wewnętrzna nie jest mierzona bezpośrednio, w środowiskach cyfrowych pozostawia wyraźny ślad behawioralny (*digital footprints*): postępujące wydłużanie przerw między logowaniami, zanik rozwiązywania quizów, unikanie forów dyskusyjnych oraz przejście z aktywności sprawdzających do biernego otwierania materiałów. Ponieważ kluczowa jest nie tylko suma kliknięć, lecz ich **sekwencja, rytm i chronologia**, proces ten modelujemy jako sekwencyjny szereg czasowy zachowań.

### 1.2. Cel praktyczny (Early Warning System)
Głównym celem wdrożeniowym projektu jest stworzenie **systemu wczesnego ostrzegania (Early Warning System)**. Na podstawie ruchomego okna obserwacji aktywności studenta z ostatnich $W$ dni ($W \in \{7, 14, 28\}$) model szacuje prawdopodobieństwo formalnego wyrejestrowania się z kursu w ciągu kolejnych **7 dni**. Horyzont 7 dni jest optymalnym kompromisem operacyjnym: jest wystarczająco bliski, by zapewnić wysoką precyzję predykcji, a jednocześnie daje tutorowi lub doradcy akademickiemu realny czas na interwencję wspierającą (np. bezpośredni kontakt mailowy, indywidualne konsultacje, uelastycznienie harmonogramu).

### 1.3. Cel naukowo-techniczny
- **Wartość informacyjna dynamiki sekwencyjnej:** weryfikacja, czy porządek chronologiczny i zmiany rytmu pracy niosą istotną wartość predykcyjną ponad proste zagregowane statystyki (np. sumaryczna liczba kliknięć).
- **Ewaluacja modeli głębokiego uczenia (LSTM vs Baseline):** porównanie skuteczności sieci rekurencyjnej LSTM (*Long Short-Term Memory*) z klasycznymi algorytmami uczenia maszynowego (Random Forest, XGBoost) operującymi na cechach zbiorczych z tego samego okna.
- **Wyjaśnialność i rekonstrukcja trajektorii:** zastosowanie metod interpretowalności (mechanizm uwagi – Attention, wartości SHAP) w celu wyodrębnienia i formalnego opisania mierzalnego wzorca behawioralnego poprzedzającego decyzję o rezygnacji.

---

## 2. Wykonane działania w ramach Etapu 1

- **Sformułowanie tematu i pytań badawczych:** określono precyzyjne ramy zadania jako binarnej klasyfikacji sekwencji z ruchomym oknem obserwacji i horyzontem 7-dniowym.
- **Postawienie i sformalizowanie hipotez badawczych:** zdefiniowano trzy hipotezy (H1: sygnały behawioralne, H2: przewaga LSTM nad XGBoost, H3: 3-etapowy wzorzec kryzysu) wraz z precyzyjną metodologią ich statystycznej weryfikacji.
- **Szczegółowa weryfikacja bazy OULAD:** przeanalizowano strukturę 7 tabel, relacje kluczy obcych oraz format zapisu czasu, potwierdzając pełną kompatybilność zbioru z wymaganiami zadania.
- **Opracowanie rygoru metodologicznego:** zaprojektowano podział danych na poziomie studenta (*Group K-Fold* po `id_student`) oraz mechanizm *Sliding Window* całkowicie wykluczający wyciek danych z przyszłości (*Data Leakage*).
- **Organizacja środowiska i podział ról:** skonfigurowano repozytorium GitHub, przygotowano skrypt automatycznego pobierania danych oraz rozdzielono role i zadania w zespole.

---

## 3. Pytania badawcze i hipotezy

### 3.1. Pytania badawcze
- **PB1:** Czy dynamika zmian w dziennej sekwencji interakcji studenta z LMS (liczba kliknięć w poszczególne typy zasobów, długość przerw bez aktywności, dynamika ocen) pozwala przewidzieć wyrejestrowanie z kursu z 7-dniowym wyprzedzeniem?
- **PB2:** Czy sieć sekwencyjna LSTM osiąga istotnie wyższą skuteczność (PR-AUC, F1-Score) niż klasyczne modele drzewiaste (Random Forest, XGBoost) operujące na cechach zagregowanych z tego samego okna obserwacji?
- **PB3:** Jakie sekwencje zachowań (np. zanik aktywności interaktywnej, wydłużanie przerw, spadek wyników) stanowią najsilniejszy mierzalny sygnał ostrzegawczy przed porzuceniem kursu?
- **PB4:** W jaki sposób długość okna obserwacji ($W \in \{7, 14, 28\}$ dni) wpływa na jakość, stabilność i wrażliwość predykcji modelu LSTM?

### 3.2. Hipotezy badawcze i metodologia weryfikacji

| Hipoteza | Treść hipotezy | Metodologia weryfikacji empirycznej |
| :--- | :--- | :--- |
| **H1 (sygnały behawioralne)** | Wydłużenie przerw (liczby dni bez aktywności w VLE) połączone ze spadkiem interakcji z zasobami sprawdzającymi (quizy, zadania) w oknie obserwacji jest statystycznie istotnym predyktorem formalnego wyrejestrowania się z kursu w ciągu najbliższych 7 dni. | Porównanie rozkładów cech w grupie rezygnujących i kontynuujących nieparametrycznym testem Manna-Whitneya ($\alpha = 0{,}05$ z korektą na wielokrotne testowanie) wraz z estymacją wielkości efektu (Cliff’s delta / rank-biserial). Uzupełnienie: analiza istotności cech (*Feature Importance*) w modelach drzewiastych. |
| **H2 (przewaga sieci LSTM)** | Dzięki zdolności do modelowania zależności czasowych i chronologii zdarzeń, model LSTM uzyska wyższe PR-AUC oraz F1-Score (o co najmniej 5 punktów procentowych) niż najlepszy model klasyczny (XGBoost) bazujący na cechach zagregowanych. | Strojenie hiperparametrów z użyciem Group K-Fold (grupowanie po `id_student`). Ostateczna ocena na wydzielonym jednorazowo zbiorze testowym. Wyznaczenie 95% przedziałów ufności metodą bootstrapingu na poziomie studentów. Identyczne podziały danych i okna dla wszystkich modeli. |
| **H3 (struktura kryzysu)** | Analiza interpretowalności modelu (wagi uwagi Attention / SHAP) ujawni powtarzalny, 3-etapowy wzorzec wycofywania się: (1) spadek aktywności interaktywnej przy sporadycznym otwieraniu materiałów pasywnych, (2) wydłużenie przerw powyżej wyznaczonego empirycznie progu (wstępnie $\ge 3$ dni), (3) całkowity zanik interakcji przed wyrejestrowaniem. | Analiza rozkładu ważności kroków czasowych dla predykcji pozytywnych ($Y = 1$), klasteryzacja wektorów ukrytych stanów sieci LSTM (lub trajektorii cech) oraz weryfikacja stabilności wzorca pomiędzy różnymi modułami kursów. |

---

## 4. Analiza danych OULAD, specyfika osi czasu i uzasadnienie adekwatności

### 4.1. Struktura bazy OULAD
Baza OULAD składa się z 7 powiązanych relacyjnie tabel w formacie CSV. Student jest jednoznacznie identyfikowany trójką: `(code_module, code_presentation, id_student)`.

| Plik CSV | Zawartość i rola w projekcie | Kluczowe zmienne |
| :--- | :--- | :--- |
| `studentVle.csv` | Dziennik interakcji z VLE (10 655 280 wierszy). Zawiera sumaryczną liczbę kliknięć studenta w dany materiał w danym dniu. Stanowi podstawę konstrukcji sekwencji czasowych dla LSTM. | `id_student`, `id_site`, `date`, `sum_click`, `code_module`, `code_presentation` |
| `vle.csv` | Katalog materiałów dydaktycznych. Umożliwia sklasyfikowanie każdego `id_site` jako zasobu interaktywnego (quiz, forumng, subpage) lub pasywnego (oucontent, url, resource). | `id_site`, `activity_type`, `week_from`, `week_to` |
| `studentRegistration.csv` | Daty zapisu i wyrejestrowania z kursu. Stanowi bezpośrednie źródło etykiety predykcyjnej $Y$. Wartość NA w `date_unregistration` oznacza brak rezygnacji. | `id_student`, `date_registration`, `date_unregistration` |
| `studentAssessment.csv` & `assessments.csv` | Rejestr ocen i terminów oddania prac (TMA – Tutor Marked, CMA – Computer Marked, Exam). Umożliwia zasilenie wektora cech wynikami cząstkowymi studenta. | `id_assessment`, `id_student`, `date_submitted`, `score`, `date`, `weight`, `assessment_type` |
| `studentInfo.csv` | Dane demograficzne (wiek, wykształcenie, płeć) oraz zagregowany wynik końcowy modułu (Pass, Fail, Withdrawn, Distinction). Wykorzystywane do walidacji. | `id_student`, `highest_education`, `age_band`, `disability`, `final_result` |
| `courses.csv` | Metadane edycji modułów, w tym długość trwania prezentacji kursu w dniach. | `code_module`, `code_presentation`, `module_presentation_length` |

### 4.2. Reprezentacja czasu w OULAD: dlaczego brak bezwzględnych timestampów (hh:mm:ss)?
W zbiorze OULAD czas nie jest zapisany w formacie stempla UNIX (rok-miesiąc-dzień godzina:minuta:sekunda), lecz w postaci **znormalizowanej, dyskretnej osi czasu o ziarnistości dziennej (zmienna `date` $\in \mathbb{Z}$)**. Zmienna ta określa liczbę dni względem oficjalnego startu modułu (dzień 0). Wartości ujemne (np. `date = -10`) oznaczają aktywność w fazie prerejestracji, natomiast wartości dodatnie reprezentują kolejne dni trwania kursu.

Brak dokładnych sygnatur godzinowych wynika z dwóch kluczowych, uzasadnionych przyczyn:
1. **Anonimizacja i ochrona prywatności (RODO / UK Data Protection Act):** Open University celowo usunął dokładne godziny i minuty zdarzeń. W połączeniu z danymi demograficznymi dokładne sygnatury czasowe umożliwiłyby profilowanie i deanonimizację poszczególnych osób na podstawie ich rytmu dobowego oraz lokalnych stref czasowych.
2. **Eliminacja szumu mikro-interakcji:** W badaniach nad procesem uczenia się milisekundowe odstępy między kliknięciami w obrębie pojedynczej sesji (np. kilkukrotne odświeżenie strony, przechodzenie między kartami) wprowadzają szum stochastyczny, nie niosąc informacji o stanie wiedzy ani motywacji studenta.

### 4.3. Merytoryczne uzasadnienie adekwatności OULAD do modelowania sieciami LSTM
Brak sygnatur sub-dziennych nie tylko nie stanowi przeszkody badawczej, lecz w kontekście modelowania sekwencji sieciami LSTM jest **podejściem metodologicznie i obliczeniowo optymalnym**:
1. **Światowy standard (Gold Standard Benchmark) w literaturze naukowej:** Zbiór OULAD (Kuzilek, Hlosta, Zdrahal, *Nature Scientific Data* 2017) jest najszerzej stosowanym zbiorem odniesienia w obszarach Educational Data Mining (EDM) oraz Learning Analytics. Setki recenzowanych artykułów (m.in. w IEEE Transactions on Learning Technologies, Computers & Education, LAK) wykorzystują dokładnie tę dzienną granulację OULAD do prognozowania rezygnacji i modelowania trajektorii przy użyciu sieci LSTM, GRU i architektur z mechanizmem uwagi.
2. **Matematyczne dopasowanie do horyzontu pamięci LSTM:** Gdyby dane rejestrowano na poziomie pojedynczych kliknięć co do sekundy, trajektoria jednego studenta w semestrze liczyłaby 30 000 – 70 000 kroków czasowych. Przy takich długościach sieci rekurencyjne cierpią na katastrofalny zanik gradientu (*vanishing gradient*) oraz eksplozję zapotrzebowania na pamięć GPU. Reprezentacja dzienna w oknie $W \in \{7, 14, 28\}$ dni tworzy sekwencje o długości kilkunastu kroków – jest to idealny, stabilny numerycznie zakres efektywnej pamięci (*effective memory horizon*) komórek LSTM.
3. **Bezpośrednia reprezentacja przerw (inactivity gaps):** Dyskretna oś dzienna pozwala na elementarne i precyzyjne zdefiniowanie cechy kluczowej dla hipotez H1 i H3 – liczby kolejnych dni bez logowania (dni bezczynności). Każdy dzień bez wpisu w `studentVle` bezpośrednio inkrementuje licznik przerwy.
4. **Spójność z realnym celem interwencyjnym (Early Warning System):** Predykcja wyprzedzająca o 7 dni to zadanie w skali makro. Działania zaradcze tutora (kontakt ze studentem, przygotowanie materiałów wyrównawczych) odbywają się w cyklu dni i tygodni, a nie sekund. Dyskretna oś dzienna idealnie odwzorowuje dynamikę tego procesu.

---

## 5. Formalizacja zadania predykcyjnego (Sliding Window)

Zadanie definiujemy jako **binarną klasyfikację wielowymiarowego szeregu czasowego** z wykorzystaniem techniki przesuwnego okna obserwacji:
- **Okno obserwacji ($W$):** Dla wybranego dnia kursu $T$ wejściem modelu jest macierz cech $X_T$ z przedziału $[T-W+1, T]$ o wymiarze $W \times K$, gdzie $K$ to liczba cech dziennych. Testowane będą okna $W \in \{7, 14, 28\}$ dni. Dla początkowych dni kursu stosowane jest dopełnianie zerami (*zero-padding*) z maskowaniem (*masking*).
- **Wektor cech dziennych ($x_t$):** W każdym dniu $t$ okna wektor cech obejmuje: (1) liczbę kliknięć w zasoby interaktywne/sprawdzające (quiz, forumng, ouwiki, subpage), (2) liczbę kliknięć w zasoby pasywne (oucontent, url, resource), (3) binarny znacznik aktywności w danym dniu, (4) długość bieżącej przerwy (liczba dni od ostatniej aktywności) oraz (5) skumulowaną średnią ważoną ocen z zadań oddanych do dnia $T$ (`date_submitted` $\le T$), co bezwzględnie zabezpiecza przed wyciekiem danych z przyszłości.
- **Zmienna docelowa ($Y_T$):** $Y_T = 1$, jeśli student formalnie wyrejestrował się w oknie predykcyjnym (`date_unregistration` $\in (T, T+7]$), w przeciwnym wypadku $Y_T = 0$.
- **Warunki generowania próbek:** Okna generowane są wyłącznie dla dni, w których student posiadał status aktywnego uczestnika (`date_registration` $\le T <$ `date_unregistration`). Rejestracje z wyrejestrowaniem przed startem modułu (`date_unregistration` $< 0$) są odrzucane z uwagi na brak historii aktywności w VLE.

---

## 6. Identyfikacja ryzyk, ograniczeń i metody mitygacji

| Zidentyfikowane ryzyko / ograniczenie | Wpływ na rzetelność projektu | Przyjęta strategia mitygacji |
| :--- | :--- | :--- |
| **Wyciek danych (Data Leakage) między oknami oraz z przyszłości** | Sztucznie zawyżone metryki skuteczności i brak zdolności generalizacji modelu w warunkach rzeczywistych. | Rygorystyczny podział zbiorów (70% train, 15% val, 15% test) i Group K-Fold wyłącznie na poziomie unikalnego `id_student`. Cechy i oceny liczone wyłącznie na podstawie danych do dnia $T$. |
| **Ekstremalna nierównowaga klas (ok. 1% okien z $Y = 1$)** | Trywialny model przewidujący zawsze klasę dominującą ($Y = 0$) uzyskuje ~99% dokładności (Accuracy), będąc całkowicie bezużytecznym. | Całkowite odrzucenie Accuracy jako metryki optymalizacyjnej. Wykorzystanie PR-AUC (*Precision-Recall AUC*) oraz F1-Score jako metryk głównych. W uczeniu LSTM: ważona funkcja straty (*Weighted BCE*) lub Focal Loss. |
| **Brak przewagi LSTM nad modelami klasycznymi (odrzucenie H2)** | Podważenie zasadności stosowania modeli głębokiego uczenia w zadaniu. | Rzetelne zaraportowanie wyników empirycznych. Zastosowanie wzbogaconej inżynierii cech zagregowanych w podoknach dla modeli XGBoost/Random Forest. |
| **Pasywne porzucenie (cichy dropout) bez formalnego wyrejestrowania** | Część studentów zaprzestaje nauki, lecz nie składa formalnej rezygnacji (etykieta $Y = 0$ pomimo faktycznego porzucenia). | Jawne wskazanie tego zjawiska jako formalnego ograniczenia badania. Dodatkowa analiza korelacji z polem `final_result` oraz modelowanie alternatywnej etykiety opartej na braku aktywności przez 28 dni. |
| **Wysoki wolumen logów interakcji (ponad 10,6 mln rekordów)** | Długi czas przetwarzania, wąskie gardła pamięci RAM podczas generowania sekwencji. | Wstępna agregacja logów do formatu Parquet na poziomie student-dzień-kategoria zasobu. Prototypowanie na próbce reprezentatywnej dwóch modułów przed finalnym treningiem. |

---

## 7. Organizacja pracy i podział ról w zespole

| Rola w zespole | Osoba odpowiedzialna | Szczegółowy zakres zadań i odpowiedzialności |
| :--- | :--- | :--- |
| **Sekretarz** | Patrycja Goźlińska | Prowadzenie dokumentacji projektowej, nadzór nad terminowością i formalną strukturą cotygodniowych raportów, systematyzacja wniosków i przygotowanie materiałów prezentacyjnych. |
| **Badacz** | Katarzyna Rogozińska | Eksploracyjna analiza danych (EDA), weryfikacja założeń statystycznych (test Manna-Whitneya dla H1), ekstrakcja i interpretacja cech behawioralnych (analiza kryzysu motywacyjnego za pomocą Attention/SHAP dla H3). |
| **Moderator** | Weronika Bieńkowska | Opracowanie architektury modeli AI (konstrukcja i trenowanie sieci LSTM w PyTorch oraz modeli referencyjnych XGBoost/Random Forest dla H2), implementacja potoku Sliding Window, koordynacja repozytorium GitHub. |

---

## 8. Plan działań na kolejny etap (Zajęcia 3–4: Pozyskanie danych)

Zgodnie z harmonogramem przedmiotu, kolejny etap prac obejmuje przygotowanie kompletnego raportu z pozyskania danych (Raport na zajęcia 4). Zaplanowano następujące kroki:
1. **Walidacja integralności relacyjnej:** weryfikacja spójności kluczy obcych łączących tabele `studentVle`, `vle`, `assessments` oraz `studentRegistration`, ze szczególnym uwzględnieniem identyfikatorów z wielokrotnymi rejestracjami.
2. **Analiza braków danych i poprawności:** szczegółowy audyt brakujących wartości (np. `date_unregistration`, `score` w ocenach) oraz analiza rekordów z ujemnymi wartościami `date` (prerejestracja).
3. **Konstrukcja zautomatyzowanego pipeline'u ekstrakcji cech:** przygotowanie skryptów w Pythonie przetwarzających logi `studentVle` do dziennych wektorów cech z podziałem na aktywności interaktywne i pasywne.
4. **Opracowanie reprezentatywnych wizualizacji wstępnych:** wykresy rozkładu przerw w logowaniach dla studentów rezygnujących vs zdających kurs na potrzeby raportu na zajęcia 4.

---

### Bibliografia i źródło danych:
*Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. Scientific Data, 4, 170171. DOI: 10.1038/sdata.2017.171 (Licencja CC BY 4.0).*
