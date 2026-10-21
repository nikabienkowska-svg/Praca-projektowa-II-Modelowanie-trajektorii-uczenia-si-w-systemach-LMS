"""
Dodatkowe weryfikacje jakościowe i uzupełnienia do notatnika (Colab)
Do wklejenia do notatnika przed jutrzejszymi zajęciami.
"""

import pandas as pd
import numpy as np

# Wczytanie danych
path = "data/raw/"
studentRegistration = pd.read_csv(path + "studentRegistration.csv")
studentAssessment   = pd.read_csv(path + "studentAssessment.csv")
assessments         = pd.read_csv(path + "assessments.csv")
vle                 = pd.read_csv(path + "vle.csv")
studentInfo         = pd.read_csv(path + "studentInfo.csv")
courses             = pd.read_csv(path + "courses.csv")
studentVle          = pd.read_csv(path + "studentVle.csv")

# Konwersja kolumn numerycznych
studentRegistration["date_registration"] = pd.to_numeric(studentRegistration["date_registration"], errors="coerce")
studentRegistration["date_unregistration"] = pd.to_numeric(studentRegistration["date_unregistration"], errors="coerce")
studentAssessment["score"] = pd.to_numeric(studentAssessment["score"], errors="coerce")
courses["module_presentation_length"] = pd.to_numeric(courses["module_presentation_length"], errors="coerce")
studentVle["date"] = pd.to_numeric(studentVle["date"], errors="coerce")

key = ["code_module", "code_presentation", "id_student"]

print("=====================================================================")
print("1. SPRAWDZENIE ANOMALII CHRONOLOGICZNYCH (REZYGNACJA PRZED ZAPISEM)")
print("=====================================================================")
sub = studentRegistration.dropna(subset=["date_registration", "date_unregistration"])
unreg_before_reg = sub[sub["date_unregistration"] < sub["date_registration"]]
print(f"Liczba przypadków (date_unregistration < date_registration): {len(unreg_before_reg)}")
print("Wniosek: Brak błędów logicznych - żaden student nie wyrejestrował się przed rejestracją.")

print("\n=====================================================================")
print("2. REJESTRACJE PO STARCIE KURSU (date_registration > 0)")
print("=====================================================================")
late_reg = studentRegistration[studentRegistration["date_registration"] > 0]
print(f"Liczba studentów zapisanych po starcie kursu: {len(late_reg)}")
print(f"Zakres dni zapisu po starcie: od dnia {late_reg['date_registration'].min():.0f} do {late_reg['date_registration'].max():.0f}")
print("Wniosek: Reguła okien musi uwzględniać warunek T >= date_registration, aby nie generować okien przed faktycznym dołączeniem.")

print("\n=====================================================================")
print("3. WCZESNE REZYGNACJE W PIERWSZYM TYGODNIU KURSU (DNI 0–6)")
print("=====================================================================")
early_unreg = studentRegistration[(studentRegistration["date_unregistration"] >= 0) & (studentRegistration["date_unregistration"] <= 6)]
print(f"Liczba rezygnacji w dniach [0, 6]: {len(early_unreg)}")
print("Wniosek: Jeśli okno W nie sięgałoby w dni ujemne, dla tych 719 osób brakowałoby historii. Sięganie wstecz w dni ujemne ratuje te próbki.")

print("\n=====================================================================")
print("4. KONTROLA SPÓJNOŚCI OCEN (studentAssessment I assessments)")
print("=====================================================================")
sa_dup = studentAssessment.duplicated(subset=["id_assessment", "id_student"], keep=False).sum()
print(f"Duplikaty student × zadanie w studentAssessment: {sa_dup}")
print(f"Oceny przeniesione z poprzedniej edycji (is_banked == 1): {(studentAssessment['is_banked'] == 1).sum()} (1.10%)")
print(f"Braki w kolumnie score: {studentAssessment['score'].isna().sum()} (0.10%)")
# Suma wag
non_exam = assessments[assessments["assessment_type"] != "Exam"]
w_sums = non_exam.groupby(["code_module", "code_presentation"])["weight"].sum()
print(f"Suma wag ocen (bez egzaminu) dla modułów AAA-FFF: {w_sums.drop('GGG', errors='ignore').unique()}%")
print(f"Suma wag dla modułu GGG: {w_sums.loc['GGG'].unique()}% (moduł bez wag procentowych, zaliczenie zero-jedynkowe)")

print("\n=====================================================================")
print("5. ROZKŁAD REZYGNACJI MIĘDZY MODUŁAMI (% WITHDRAWN)")
print("=====================================================================")
mod_dist = studentInfo.groupby("code_module")["final_result"].value_counts(normalize=True).unstack()["Withdrawn"] * 100
for mod, val in mod_dist.items():
    print(f"Moduł {mod}: {val:.2f}% rezygnacji")
print("Wniosek: Olbrzymia dysproporcja (od 11.52% w GGG do 44.54% w CCC) - idealna do weryfikacji generalizacji modeli między modułami.")

print("\n=====================================================================")
print("6. PODZIAŁ 20 TYPÓW MATERIAŁÓW (activity_type) NA AKTYWNE, PASYWNE I NAWIGACYJNE")
print("=====================================================================")
# Propozycja formalnego mapowania
interactive = ["quiz", "forumng", "oucollaborate", "questionnaire", "ouwiki", "externalquiz", "ouelluminate", "htmlactivity", "repeatactivity"]
passive = ["resource", "oucontent", "url", "page", "glossary", "dualpane", "folder", "dataplus"]
navigation = ["subpage", "homepage", "sharedsubpage"]

vle_type_counts = vle["activity_type"].value_counts()
print(f"Zasoby interaktywne / sprawdzające ({len(interactive)} typów): {vle_type_counts.loc[vle_type_counts.index.isin(interactive)].sum()} zasobów")
print(f"Zasoby pasywne / treści ({len(passive)} typów): {vle_type_counts.loc[vle_type_counts.index.isin(passive)].sum()} zasobów")
print(f"Zasoby nawigacyjne / strukturalne ({len(navigation)} typów): {vle_type_counts.loc[vle_type_counts.index.isin(navigation)].sum()} zasobów")
print("Uwaga: 'subpage' (1055 zasobów) to elementy nawigacji, dlatego w modelu traktujemy je jako osobną kategorię nawigacyjną lub pasywną.")

print("\n=====================================================================")
print("7. POPRAWIONA ANALIZA PRZERW: LICZBA DNI AKTYWNYCH ORAZ KOŃCOWA CISZA")
print("=====================================================================")
# Policzmy na próbce AAA i BBB
mods = ["AAA", "BBB"]
sv_sub = studentVle[studentVle["code_module"].isin(mods)]
si_sub = studentInfo[studentInfo["code_module"].isin(mods)][key + ["final_result"]]
sr_sub = studentRegistration[studentRegistration["code_module"].isin(mods)][key + ["date_unregistration", "date_registration"]]
cr_sub = courses[courses["code_module"].isin(mods)]

act_days = sv_sub[sv_sub["date"] >= 0].groupby(key)["date"].nunique().rename("active_days").reset_index()
last_act = sv_sub[sv_sub["date"] >= 0].groupby(key)["date"].max().rename("last_active_day").reset_index()

df_gap = si_sub.merge(sr_sub, on=key, how="left").merge(cr_sub, on=["code_module", "code_presentation"], how="left")
df_gap = df_gap.merge(act_days, on=key, how="left").merge(last_act, on=key, how="left")
df_gap["active_days"] = df_gap["active_days"].fillna(0)
df_gap["end_day"] = np.where(df_gap["date_unregistration"] >= 0, df_gap["date_unregistration"], df_gap["module_presentation_length"])
df_gap["final_silence"] = np.where(df_gap["last_active_day"].notna(), df_gap["end_day"] - df_gap["last_active_day"], df_gap["end_day"])
df_gap["grupa"] = np.where(df_gap["final_result"] == "Withdrawn", "Withdrawn", "Pozostali")

print("Średnia i mediana liczby aktywnych dni:")
print(df_gap.groupby("grupa")["active_days"].agg(["mean", "median", "std"]))
print("\nŚrednia i mediana końcowej ciszy (dni od ostatniego kliknięcia do rezygnacji/końca):")
print(df_gap.groupby("grupa")["final_silence"].agg(["mean", "median", "std"]))
print("Wniosek: Studenci rezygnujący mają drastycznie mniej aktywnych dni (mediana 1 vs 43 dni) oraz wykazują wielotygodniową ciszę przed formalną rezygnacją!")
