"""Skrypt przygotowujący i filtrujący zbiór Webis-CMV-20 (ChangeMyView).
Ekstrahuje pary dyskusji: post inicjujący (OP) vs argument perswazyjny (delta) vs kontrargument bez delty.
"""

import bz2
import json
import os
import argparse
from typing import List, Dict, Any


def extract_comment_text(comment_obj: Dict[str, Any]) -> str:
    """Ekstrahuje scalony tekst wypowiedzi z listy komentarzy."""
    comments = comment_obj.get("comments", [])
    texts = [c.get("body", "") for c in comments if c.get("body")]
    return "\n\n".join(texts)


def prepare_webis_cmv_dataset(
    raw_pairs_path: str = "data/raw/pairs.jsonl.bz2",
    output_path: str = "data/processed/cmv_persuasion_pairs_sample.jsonl",
    max_records: int = 1000,
):
    """Przetwarza skompresowany plik pairs.jsonl.bz2 do czystego, lekkiego formatu JSONL."""
    if not os.path.exists(raw_pairs_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {raw_pairs_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Przetwarzanie {raw_pairs_path} -> {output_path} (maks. {max_records} rekordów)...")

    processed_count = 0
    with bz2.open(raw_pairs_path, "rt", encoding="utf-8") as in_f, open(output_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            
            item = json.loads(line)
            sub = item.get("submission", {})
            title = sub.get("title", "").strip()
            selftext = sub.get("selftext", "").strip()
            sub_id = item.get("submission_id", "")
            author = sub.get("author", "")

            # Tekst argumentu, który przekonał autora (Delta - zmiana zdania)
            delta_body = extract_comment_text(item.get("delta_comment", {}))
            # Tekst argumentu, który NIE przekonał autora
            nodelta_body = extract_comment_text(item.get("nodelta_comment", {}))

            if not (title and delta_body and nodelta_body):
                continue

            record = {
                "submission_id": sub_id,
                "author": author,
                "title": title,
                "op_text": selftext,
                "delta_argument": delta_body,
                "nodelta_argument": nodelta_body,
                "similarity_score": item.get("comments_similarity", None),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_count += 1

            if processed_count >= max_records:
                break

    print(f"✅ Zapisano pomyślnie {processed_count} czystych rekordów w {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_records", type=int, default=1000)
    args = parser.parse_args()
    prepare_webis_cmv_dataset(max_records=args.max_records)
