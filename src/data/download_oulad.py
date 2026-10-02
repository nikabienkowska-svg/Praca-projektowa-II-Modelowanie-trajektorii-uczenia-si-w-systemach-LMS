"""Skrypt wspomagający pobieranie lub generowanie syntetycznej próbki danych OULAD
do szybkich testów lokalnych."""

import os
import argparse
import urllib.request
import zipfile
import pandas as pd
import numpy as np


def generate_mock_oulad_sample(output_dir: str = "data/raw", num_students: int = 200):
    """Generuje spójną, syntetyczną minipróbkę tabel OULAD do testów pipeline'u bez pobierania całego zbioru."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    print(f"Generowanie syntetycznej próbki OULAD ({num_students} studentów) w {output_dir}...")

    # 1. courses.csv
    courses_df = pd.DataFrame({
        "code_module": ["AAA", "BBB", "CCC"],
        "code_presentation": ["2013J", "2013J", "2014J"],
        "module_presentation_length": [268, 268, 269],
    })
    courses_df.to_csv(os.path.join(output_dir, "courses.csv"), index=False)

    # 2. assessments.csv
    assessments_df = pd.DataFrame({
        "code_module": ["AAA"] * 5 + ["BBB"] * 5,
        "code_presentation": ["2013J"] * 10,
        "id_assessment": list(range(1001, 1011)),
        "assessment_type": ["TMA", "TMA", "CMA", "CMA", "Exam"] * 2,
        "date": [30, 60, 90, 120, 240, 28, 56, 84, 119, 240],
        "weight": [10.0, 20.0, 10.0, 10.0, 50.0] * 2,
    })
    assessments_df.to_csv(os.path.join(output_dir, "assessments.csv"), index=False)

    # 3. vle.csv
    activity_types = ["oucontent", "forumng", "quiz", "resource", "subpage", "url"]
    vle_rows = []
    vle_id = 10001
    for mod in ["AAA", "BBB", "CCC"]:
        for pres in ["2013J", "2014J"]:
            for act in activity_types:
                for i in range(5):
                    vle_rows.append({
                        "id_site": vle_id,
                        "code_module": mod,
                        "code_presentation": pres,
                        "activity_type": act,
                        "week_from": None,
                        "week_to": None,
                    })
                    vle_id += 1
    vle_df = pd.DataFrame(vle_rows)
    vle_df.to_csv(os.path.join(output_dir, "vle.csv"), index=False)

    # 4. studentInfo.csv & studentRegistration.csv
    student_ids = list(range(100001, 100001 + num_students))
    modules = np.random.choice(["AAA", "BBB"], size=num_students)
    presentations = ["2013J"] * num_students
    genders = np.random.choice(["M", "F"], size=num_students)
    education = np.random.choice(["HE Qualification", "A Level", "Lower Than A Level"], size=num_students)
    age_bands = np.random.choice(["0-35", "35-55", "55<="], size=num_students)
    
    # 25% dropout rate
    dropout_mask = np.random.rand(num_students) < 0.25
    final_results = []
    unreg_dates = []
    for is_dropout in dropout_mask:
        if is_dropout:
            final_results.append("Withdrawn")
            unreg_dates.append(int(np.random.randint(20, 180)))
        else:
            final_results.append(np.random.choice(["Pass", "Distinction", "Fail"], p=[0.6, 0.2, 0.2]))
            unreg_dates.append(np.nan)

    student_info_df = pd.DataFrame({
        "code_module": modules,
        "code_presentation": presentations,
        "id_student": student_ids,
        "gender": genders,
        "region": "East Anglian Region",
        "highest_education": education,
        "imd_band": "90-100%",
        "age_band": age_bands,
        "num_of_prev_attempts": 0,
        "studied_credits": 60,
        "disability": "N",
        "final_result": final_results,
    })
    student_info_df.to_csv(os.path.join(output_dir, "studentInfo.csv"), index=False)

    student_reg_df = pd.DataFrame({
        "code_module": modules,
        "code_presentation": presentations,
        "id_student": student_ids,
        "date_registration": np.random.randint(-30, 0, size=num_students),
        "date_unregistration": unreg_dates,
    })
    student_reg_df.to_csv(os.path.join(output_dir, "studentRegistration.csv"), index=False)

    # 5. studentVle.csv (sekwencja interakcji dziennych)
    vle_ids_available = vle_df["id_site"].values
    vle_records = []
    for s_id, is_dropout, unreg_d in zip(student_ids, dropout_mask, unreg_dates):
        max_active_day = int(unreg_d) if is_dropout else 200
        # Aktywni studenci logują się co kilka dni
        active_days = np.random.choice(range(0, max_active_day), size=min(max_active_day, 40), replace=False)
        for d in active_days:
            num_sites = np.random.randint(1, 5)
            sites = np.random.choice(vle_ids_available, size=num_sites)
            for site in sites:
                vle_records.append({
                    "code_module": "AAA",
                    "code_presentation": "2013J",
                    "id_student": s_id,
                    "id_site": site,
                    "date": int(d),
                    "sum_click": int(np.random.randint(1, 20)),
                })
    student_vle_df = pd.DataFrame(vle_records)
    student_vle_df.to_csv(os.path.join(output_dir, "studentVle.csv"), index=False)

    # 6. studentAssessment.csv
    assessment_records = []
    for s_id in student_ids:
        for a_id in assessments_df["id_assessment"].values[:3]:
            assessment_records.append({
                "id_assessment": a_id,
                "id_student": s_id,
                "date_submitted": int(np.random.randint(25, 95)),
                "is_banked": 0,
                "score": float(np.random.randint(40, 100)),
            })
    pd.DataFrame(assessment_records).to_csv(os.path.join(output_dir, "studentAssessment.csv"), index=False)

    print("✅ Pomyślnie wygenerowano kompletny zestaw testowy OULAD w folderze:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pobieranie lub generowanie danych OULAD.")
    parser.add_argument("--mock", action="store_true", help="Wygeneruj lokalną próbkę testową (mock)")
    parser.add_argument("--num_students", type=int, default=200, help="Liczba studentów w próbce testowej")
    args = parser.parse_args()

    if args.mock:
        generate_mock_oulad_sample(num_students=args.num_students)
    else:
        print("Aby pobrać pełny zbiór OULAD (ok. 450 MB), odwiedź:")
        print("https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/")
        print("i rozpakuj pliki CSV do katalogu data/raw/")
        print("Możesz także uruchomić: python src/data/download_oulad.py --mock")
