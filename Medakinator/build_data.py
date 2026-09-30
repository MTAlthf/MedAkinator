#!/usr/bin/env python3
"""
build_data.py
=============
Mengubah dataset mentah (CSV) menjadi model probabilistik MedAkinator yang
kompak, lalu menyuntikkannya ke dalam medakinator_interface.html sehingga
halaman tersebut bisa berjalan sepenuhnya di browser (tanpa server/backend).

Alur:
  1. Pakai fungsi preprocessing & MedAkinator dari medakinator.py (CLI)
     untuk membangun matriks P(gejala | penyakit) dan prior tiap penyakit.
  2. Kuantisasi P dan prior ke uint8 (0-255) agar ukurannya kecil saat
     disimpan sebagai base64 di dalam file HTML.
  3. Ganti placeholder __MED_DATA_JSON__ di medakinator_interface.html
     dengan data JSON tersebut -> hasil akhir: medakinator.html (siap pakai).

Jalankan:
    python3 build_data.py --csv "Disease and symptoms dataset.csv"
"""

import argparse
import base64
import json
import os

import numpy as np

from medakinator import load_and_preprocess, MedAkinator


def build(csv_path: str, template_path: str, output_path: str):
    df = load_and_preprocess(csv_path)
    model = MedAkinator(df)

    # Kuantisasi ke uint8 (0-255) supaya kompak saat di-base64
    P8 = np.clip(np.round(model.P * 255), 0, 255).astype("uint8")
    prior8 = np.clip(np.round(model.prior * 255 / model.prior.max()), 1, 255).astype("uint8")

    data = {
        "diseases": model.diseases,
        "symptoms": model.symptom_names,
        "nd": len(model.diseases),
        "ns": len(model.symptom_names),
        "P_b64": base64.b64encode(P8.tobytes()).decode("ascii"),
        "prior_b64": base64.b64encode(prior8.tobytes()).decode("ascii"),
    }

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    final_html = template.replace("__MED_DATA_JSON__", json.dumps(data))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"Selesai -> {output_path} ({os.path.getsize(output_path)/1e6:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Bangun ulang medakinator.html dari dataset CSV")
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--csv", default=os.path.join(here, "Disease and symptoms dataset.csv"))
    parser.add_argument("--template", default=os.path.join(here, "medakinator_interface.html"))
    parser.add_argument("--out", default=os.path.join(here, "medakinator.html"))
    args = parser.parse_args()
    build(args.csv, args.template, args.out)


if __name__ == "__main__":
    main()