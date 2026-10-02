# Raport 1 — Zdefiniowanie problemu i hipotez badawczych
**Etap:** zajęcia 1–2 · **Temat 6:** Transformery (Attention / BERT / GPT) — Predykcja polaryzacji postawy i zmiana zdania na forach dyskusyjnych
**Zespół:** Weronika, [Członek zespołu 2], [Członek zespołu 3]
**Metoda:** Problem-Based Learning

---

## 1. Cel etapu

Wybór tematu, sformułowanie pytania badawczego i hipotez, określenie zakresu analizy oraz wstępny podział ról w zespole. Efektem ma być precyzyjnie zdefiniowany problem badawczy możliwy do weryfikacji pomiarowej na rzeczywistym korpusie tekstowym.

## 2. Wykonane działania

- Wybór tematu 6 z listy projektowej: **Predykcja polaryzacji postawy (zmiana zdania) na podstawie analizy sekwencji komentarzy na forach dyskusyjnych (Webis-CMV-20 / BERT)**.
- Doprecyzowanie problemu: analiza argumentacji na forum Reddit (subreddit r/ChangeMyView), gdzie autorzy postów przyznają nagrodę Delty ($\Delta$) komentarzom, które przekonały ich do zmiany punktu widzenia.
- Zdefiniowanie zadania jako klasyfikacji binarnej par wypowiedzi `[Post OP] + [Kontrargument]` oraz detekcji wahań w sekwencji wypowiedzi.
- Przegląd korpusu Webis-CMV-20 udostępnionego na platformie Zenodo (DOI: 10.5281/zenodo.3778298).
- Podział ról w zespole.

## 3. Uzyskane rezultaty

### Problem badawczy

W dyskusjach internetowych i na forach argumentacyjnych użytkownicy często okopują się na swoich pozycjach (polaryzacja). Trudno jest przewidzieć, jakie cechy językowe i semantyczne sprawiają, że dany argument przełamuje uprzedzenia i prowadzi do zmiany zdania (kapitulacji przekonaniowej). Otwartym pytaniem pozostaje, **czy modele kontekstowe oparte na architekturze Transformer (BERT) przewyższają klasyczne metody leksykalne i statystyczne w wykrywaniu perswazji**.

### Pytanie badawcze

> Czy model językowy oparty na architekturze Transformer (BERT / RoBERTa) pozwala na istotnie trafniejsze przewidywanie skuteczności perswazji (przyznania Delty) oraz momentu zmiany tonu wypowiedzi niż klasyczne modele bazowe oparte na cechach lingwistycznych (TF-IDF, LIWC)?

### Hipotezy badawcze

| # | Hipoteza | Sposób weryfikacji |
|---|---|---|
| **H1** | Zmianie zdania towarzyszy mierzalny wzrost wskaźnika zwrotów asekuracyjnych (*hedging words*) oraz spadek kategoryczności wypowiedzi. | Porównanie gęstości słów pewności vs wahania w wypowiedziach przed i po interakcji perswazyjnej. |
| **H2** | Model BERT uwzględniający pełny kontekst semantyczny pary post-argument osiąga istotnie wyższy wynik ROC-AUC i F1 niż klasyczny baseline. | Parowany test statystyczny porównujący model BERT z Regresją Logistyczną / Random Forest. |
| **H3** | Wagi mechanizmu atencji (Self-Attention) w modelu BERT w warstwach końcowych skupiają się na logicznych przesłankach, a nie na zabiegach czysto erystycznych. | Wizualizacja i analiza wag macierzy atencji dla wybranych par dyskusji. |

### Zakres analizy

- **Typ danych:** korpus tekstowy postów i komentarzy z r/ChangeMyView (Webis-CMV-20).
- **Zadanie główne:** klasyfikacja binarna (czy argument doprowadzi do przyznania $\Delta$).
- **Zadanie uzupełniające:** analiza ewolucji językowego wyrazu pewności w sekwencji komentarzy.
- **Metryki:** ROC-AUC, PR-AUC, F1-score, Precision, Recall.

### Podział ról (model PBL, role rotacyjne)

| Rola | Odpowiedzialność |
|---|---|
| Moderator | prowadzenie spotkań, pilnowanie terminów i tablicy zadań |
| Sekretarz | prowadzenie dokumentacji i raportów etapowych |
| Badacz / Inżynier Danych | pobieranie danych, wstępna obróbka tekstów, tokenizacja |
| Reflektor | kontrola poprawności metodologicznej i metryk ewaluacji |

## 4. Problemy i sposoby ich rozwiązania

- **Problem:** Niezbalansowanie klas — w naturalnej dyskusji argumenty zakończone zmianą zdania stanowią zaledwie ułamek procenta wszystkich komentarzy.
  *Rozwiązanie:* Wykorzystanie podzbioru `pairs.jsonl.bz2`, w którym każdy skuteczny argument z Deltą ma bezpośrednio dobrany argument kontrolny o podobnej tematyce (zbiór zbalansowany 1:1).
- **Problem:** Długość dyskusji przekraczająca limit 512 tokenów w modelu BERT.
  *Rozwiązanie:* Truncation strategii nagłówka i konkluzji (`head + tail`) lub streszczenie wstępne.

## 5. Plan działań na kolejny etap

1. Pobranie i rozpakowanie plików korpusu Webis-CMV-20.
2. Zbudowanie czystego podzbioru roboczego do analizy wstępnej (Raport 2).
3. Wyznaczenie podstawowych statystyk opisowych (liczba par, długości tekstów).
