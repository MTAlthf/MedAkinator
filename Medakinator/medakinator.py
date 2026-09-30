#!/usr/bin/env python3
"""
MedAkinator - Sistem Pakar Tebak Penyakit (versi CLI)
======================================================

Program ini "menebak" penyakit pengguna dengan cara bertanya gejala satu per satu,
mirip permainan Akinator, tetapi berbasis data medis nyata.

Alur program:
  1. PREPROCESSING   -> membaca & membersihkan dataset mentah menjadi matriks
                         biner (baris = kasus/penyakit, kolom = gejala).
  2. CORE LOGIC      -> membangun model probabilistik per penyakit, lalu di
                         setiap giliran memilih gejala paling optimal untuk
                         ditanyakan menggunakan Entropy / Information Gain
                         (prinsip yang sama dipakai algoritma Decision Tree /
                         ID3 untuk memilih atribut pemecah terbaik).
  3. CLI INTERAKTIF  -> tanya-jawab (y/n/u) di terminal. Sistem tidak dibatasi
                         maksimal 25 pertanyaan; ia akan terus menjelajahi
                         SELURUH gejala yang masih informatif di dataset, dan
                         baru berhenti bila gejala sudah habis dieksplorasi
                         atau keyakinannya sudah tinggi & tidak ambigu (lihat
                         run_cli). Di akhir, tampil 3 tebakan penyakit teratas
                         beserta tingkat keyakinannya.

Cara pakai:
    python3 medakinator.py --csv "Disease and symptoms dataset.csv"

Jika --csv tidak diisi, program akan mencari file bernama
"Disease and symptoms dataset.csv" di folder yang sama dengan script ini.
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd


# =============================================================================
# BAGIAN 1: DATA PREPROCESSING
# =============================================================================

def _clean_text(value: str) -> str:
    """Bersihkan satu string: hapus spasi berlebih & samakan huruf jadi lowercase."""
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)   # spasi ganda -> spasi tunggal
    return value


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Membaca CSV mentah dan mengubahnya menjadi matriks biner (one-hot):
        - baris   : satu kasus/penyakit
        - kolom   : 'Disease' + semua gejala unik
        - nilai   : 1 jika gejala muncul pada kasus tsb, 0 jika tidak

    Fungsi ini DINAMIS - otomatis mendeteksi dua kemungkinan bentuk data mentah:

    (A) Data SUDAH berbentuk biner/one-hot, contoh:
            diseases, gejala_1, gejala_2, gejala_3, ...
            flu,      1,        0,        1, ...
        -> tinggal dibersihkan nama kolom & nama penyakitnya.

    (B) Data berbentuk DAFTAR gejala mentah (belum one-hot), contoh:
            Disease, Symptom_1,        Symptom_2,       Symptom_3
            flu,     demam,             batuk,           " Pilek "
        -> setiap kolom gejala berisi TEKS nama gejala (bukan 0/1), sehingga
           perlu di-"lebur" (melt) lalu dipivot jadi one-hot encoding.
    """
    print(f"[1] Membaca dataset dari: {csv_path}")
    df_raw = pd.read_csv(csv_path)

    if df_raw.shape[1] < 2:
        raise ValueError("Dataset minimal harus punya 2 kolom (Disease + gejala).")

    # Kolom pertama = nama penyakit -> standarkan namanya jadi 'Disease'
    disease_col = df_raw.columns[0]
    df_raw = df_raw.rename(columns={disease_col: "Disease"})
    df_raw["Disease"] = df_raw["Disease"].apply(_clean_text)
    df_raw = df_raw[df_raw["Disease"] != ""]           # buang baris tanpa label penyakit

    symptom_cols = list(df_raw.columns[1:])

    # --- Deteksi bentuk data ------------------------------------------------
    sample = df_raw[symptom_cols].head(2000)
    unique_vals = pd.unique(sample.values.ravel())
    unique_vals_clean = {
        str(v).strip() for v in unique_vals if not (isinstance(v, float) and math.isnan(v))
    }
    is_already_binary = unique_vals_clean.issubset({"0", "1", "0.0", "1.0"})

    if is_already_binary:
        # ------------------ BENTUK (A): sudah biner ------------------------
        print("[1] Format terdeteksi: matriks biner (0/1) siap pakai.")
        # Gunakan tipe data hemat memori (int8) sejak awal - dataset gejala
        # bisa berukuran ratusan ribu baris x ratusan kolom.
        matrix = df_raw[symptom_cols].apply(
            lambda col: pd.to_numeric(col, errors="coerce").fillna(0).astype("int8")
        )
        matrix = (matrix > 0).astype("int8")

        # Bersihkan nama kolom gejala: strip spasi, lowercase, spasi ganda -> tunggal.
        # Juga buang sufiks ".1", ".2", dst yang otomatis ditambahkan pandas ketika
        # ada nama kolom asli yang duplikat (mis. "regurgitation" & "regurgitation.1"
        # sebenarnya adalah gejala yang sama).
        clean_cols = [_clean_text(re.sub(r"\.\d+$", "", c)) for c in symptom_cols]

        if len(set(clean_cols)) == len(clean_cols):
            # Kasus umum: tidak ada nama gejala yang jadi duplikat -> langsung rename
            matrix.columns = clean_cols
        else:
            # Ada duplikat nama gejala setelah dibersihkan -> gabungkan dengan OR
            matrix.columns = clean_cols
            matrix = matrix.T.groupby(level=0).max().T.astype("int8")

        binary_df = matrix
        binary_df.insert(0, "Disease", df_raw["Disease"].values)
        binary_df = binary_df.reset_index(drop=True)

    else:
        # ------------------ BENTUK (B): daftar gejala mentah ----------------
        print("[1] Format terdeteksi: daftar gejala mentah -> melakukan one-hot encoding...")
        melted = df_raw.melt(id_vars="Disease", value_vars=symptom_cols,
                              value_name="Symptom")
        melted["Symptom"] = melted["Symptom"].apply(_clean_text)
        melted = melted[melted["Symptom"] != ""]        # buang sel kosong / NaN

        melted["present"] = 1
        melted["row_id"] = melted.index  # supaya tiap baris asal tetap terpisah
        # Sebenarnya kita ingin one-hot PER BARIS ASAL, bukan per hasil melt.
        # Jadi gunakan index baris original sebagai id kasus.
        melted["case_id"] = df_raw.index.repeat(len(symptom_cols))[: len(melted)]

        binary_df = (
            melted.pivot_table(index="case_id", columns="Symptom",
                                values="present", fill_value=0, aggfunc="max")
            .reindex(df_raw.index, fill_value=0)
            .astype("int8")
        )
        binary_df.insert(0, "Disease", df_raw["Disease"].values)
        binary_df = binary_df.reset_index(drop=True)

    n_diseases = binary_df["Disease"].nunique()
    n_symptoms = binary_df.shape[1] - 1
    print(f"[1] Selesai preprocessing -> {binary_df.shape[0]} baris kasus, "
          f"{n_diseases} penyakit unik, {n_symptoms} gejala unik.\n")
    return binary_df


# =============================================================================
# BAGIAN 2: CORE LOGIC (MODEL PROBABILISTIK + ENTROPY / INFORMATION GAIN)
# =============================================================================

class MedAkinator:
    """
    Model penebak penyakit.

    Untuk tiap penyakit d dan gejala s, dihitung P(s=1 | d) dari data (dengan
    Laplace smoothing supaya tidak ada probabilitas 0% / 100% mutlak).

    Di setiap giliran tanya-jawab, sistem menghitung entropy dari distribusi
    kemungkinan penyakit saat ini, lalu memilih gejala yang paling menurunkan
    entropy tersebut (Information Gain tertinggi) jika ditanyakan berikutnya
    -- prinsip dasar yang sama dipakai algoritma Decision Tree (ID3/C4.5)
    untuk memilih atribut pemecah terbaik di setiap node.
    """

    def __init__(self, binary_df: pd.DataFrame, laplace_alpha: float = 1.0):
        self.symptom_names = list(binary_df.columns[1:])
        self.n_symptoms = len(self.symptom_names)

        # Prior tiap penyakit = proporsi kemunculannya di data
        counts_per_disease = binary_df["Disease"].value_counts()
        self.diseases = list(counts_per_disease.index)
        self.prior = (counts_per_disease / counts_per_disease.sum()).reindex(self.diseases).values

        # P(gejala=1 | penyakit) dengan Laplace smoothing
        grouped = binary_df.groupby("Disease")[self.symptom_names]
        sum_yes = grouped.sum().reindex(self.diseases)
        n_per_disease = grouped.size().reindex(self.diseases)

        alpha = laplace_alpha
        prob = (sum_yes.add(alpha)).div(n_per_disease.add(2 * alpha), axis=0)
        self.P = prob.values  # shape: (n_diseases, n_symptoms), P[d, s] = P(gejala s=1 | penyakit d)

        # State sesi tanya-jawab
        self.reset()

    def reset(self):
        self.posterior = self.prior.copy()
        self.asked_mask = np.zeros(self.n_symptoms, dtype=bool)

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def current_entropy(self) -> float:
        return self._entropy(self.posterior)

    def best_next_symptom(self):
        """
        Hitung Information Gain untuk setiap gejala yang BELUM ditanyakan,
        lalu kembalikan (index_gejala, nama_gejala, info_gain_terbesar).
        Mengembalikan None jika tidak ada lagi gejala yang informatif.
        """
        w = self.posterior  # (n_diseases,)
        H_current = self._entropy(w)

        # A[d, s] = P(penyakit d) * P(gejala s=1 | d)   -> gabungan "ya"
        A = w[:, None] * self.P
        p1 = A.sum(axis=0)                       # P(gejala s = 1) marjinal, shape (n_symptoms,)
        p0 = 1.0 - p1

        with np.errstate(divide="ignore", invalid="ignore"):
            post_yes = np.where(p1 > 1e-12, A / np.where(p1 > 1e-12, p1, 1), 0.0)
            B = w[:, None] * (1.0 - self.P)
            post_no = np.where(p0 > 1e-12, B / np.where(p0 > 1e-12, p0, 1), 0.0)

        def col_entropy(mat):
            log2mat = np.zeros_like(mat)
            mask = mat > 0
            log2mat[mask] = np.log2(mat[mask])
            terms = mat * log2mat
            return -terms.sum(axis=0)

        H_yes = col_entropy(post_yes)
        H_no = col_entropy(post_no)
        expected_H_after = p1 * H_yes + p0 * H_no
        info_gain = H_current - expected_H_after

        info_gain[self.asked_mask] = -np.inf
        # Gejala yang nyaris pasti ya/tidak untuk semua penyakit (p1~0 atau p1~1)
        # tidak akan menambah informasi apa pun -> otomatis dapat gain ~0.

        best_idx = int(np.argmax(info_gain))
        best_gain = info_gain[best_idx]
        if not np.isfinite(best_gain) or best_gain <= 1e-9:
            return None
        return best_idx, self.symptom_names[best_idx], float(best_gain)

    def update(self, symptom_idx: int, answer: str):
        """answer: 'y' (ya), 'n' (tidak), 'u' (tidak tahu)."""
        self.asked_mask[symptom_idx] = True
        if answer == "u":
            return  # tidak ada update, hanya ditandai sudah ditanyakan

        col = self.P[:, symptom_idx]
        likelihood = col if answer == "y" else (1.0 - col)
        new_posterior = self.posterior * likelihood

        total = new_posterior.sum()
        if total > 1e-12:
            self.posterior = new_posterior / total
        # jika total ~0 (kontradiksi total dgn semua penyakit), posterior lama dipertahankan
        # supaya sistem tidak "buntu".

    def top_k(self, k: int = 3):
        order = np.argsort(self.posterior)[::-1][:k]
        return [(self.diseases[i], float(self.posterior[i])) for i in order]


# =============================================================================
# BAGIAN 3: INTERAKSI TERMINAL (CLI)
# =============================================================================

def ask_yes_no_unknown(prompt: str) -> str:
    while True:
        ans = input(f"{prompt} [y = ya / n = tidak / u = tidak tahu / q = keluar]: ").strip().lower()
        if ans in ("y", "n", "u", "q"):
            return ans
        print("   -> Masukan tidak dikenali, ketik salah satu dari: y, n, u, q")


def run_cli(model: MedAkinator, max_questions: int = None, confidence_stop: float = 0.95,
            margin_stop: float = 0.15, min_questions_for_stop: int = 5):
    """
    Jalankan sesi tanya-jawab CLI.

    Tidak lagi dibatasi kaku ke 25 pertanyaan: `max_questions` default-nya
    None, yang berarti sistem boleh menjelajahi SELURUH gejala yang ada di
    dataset (model.n_symptoms) selama gejala tsb masih informatif (Information
    Gain > 0). Sesi berhenti lebih awal HANYA jika salah satu dari ini benar:
      1. Tidak ada lagi gejala yang informatif (best_next_symptom() -> None),
         artinya seluruh kemungkinan gejala relevan sudah dieksplorasi.
      2. Keyakinan penyakit #1 sudah tinggi (>= confidence_stop) DAN unggul
         jauh (margin >= margin_stop) dari penyakit #2 -- supaya sistem tidak
         berhenti prematur saat dua penyakit masih head-to-head (hasil yang
         tidak valid / meragukan).
      3. Pengguna sendiri memilih berhenti ('q').
    """
    if max_questions is None:
        max_questions = model.n_symptoms  # batas aman: seluruh gejala di dataset

    print("=" * 60)
    print(" MEDAKINATOR - Sistem Pakar Tebak Penyakit ")
    print("=" * 60)
    print("Jawab pertanyaan gejala berikut dengan sejujur mungkin.")
    print("Sistem akan terus menggali gejala selama masih membantu mempersempit")
    print("kemungkinan, agar hasil akhir lebih valid & merujuk penyakit yang tepat.\n")

    asked_count = 0
    while asked_count < max_questions:
        result = model.best_next_symptom()
        if result is None:
            print("\n(Seluruh kemungkinan gejala yang informatif sudah dieksplorasi.)")
            break

        idx, symptom_name, gain = result
        answer = ask_yes_no_unknown(f"Q{asked_count + 1}. Apakah Anda mengalami: '{symptom_name}'?")

        if answer == "q":
            print("\nSesi dihentikan oleh pengguna.")
            break

        model.update(idx, answer)
        asked_count += 1

        top2 = model.top_k(2)
        top1_prob = top2[0][1]
        margin = top1_prob - (top2[1][1] if len(top2) > 1 else 0.0)
        if asked_count >= min_questions_for_stop and top1_prob >= confidence_stop and margin >= margin_stop:
            print(f"\n(Keyakinan {top1_prob:.0%} dengan margin {margin:.0%} terhadap kandidat "
                  f"terkuat berikutnya -> menghentikan sesi tanya-jawab.)")
            break

    print("\n" + "=" * 60)
    print(" HASIL TEBAKAN - 3 KEMUNGKINAN PENYAKIT TERATAS ")
    print("=" * 60)
    for rank, (disease, prob) in enumerate(model.top_k(3), start=1):
        print(f"  {rank}. {disease.title():<40} (keyakinan: {prob:.1%})")
    print("=" * 60)
    print("Catatan: Ini adalah hasil sistem pakar berbasis data, BUKAN diagnosis")
    print("medis resmi. Silakan konsultasi ke dokter/tenaga medis profesional.")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MedAkinator - Sistem Pakar Tebak Penyakit")
    default_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "Disease and symptoms dataset.csv")
    parser.add_argument("--csv", default=default_csv,
                         help="Path ke file CSV dataset penyakit & gejala")
    parser.add_argument("--max-questions", type=int, default=None,
                         help="Jumlah maksimum pertanyaan per sesi. Default: None -> "
                              "jelajahi seluruh gejala di dataset (tidak dibatasi 25).")
    parser.add_argument("--confidence-stop", type=float, default=0.95,
                         help="Ambang keyakinan (0-1) penyakit teratas untuk berhenti lebih awal (default: 0.95)")
    parser.add_argument("--margin-stop", type=float, default=0.15,
                         help="Selisih minimum keyakinan antara penyakit #1 dan #2 agar boleh "
                              "berhenti lebih awal, supaya hasil tidak ambigu (default: 0.15)")
    parser.add_argument("--min-questions", type=int, default=5,
                         help="Jumlah pertanyaan minimum sebelum sistem boleh berhenti lebih awal (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: File dataset tidak ditemukan di '{args.csv}'.")
        print("Gunakan argumen --csv \"path/ke/file.csv\" untuk menunjuk lokasi file.")
        sys.exit(1)

    binary_df = load_and_preprocess(args.csv)
    model = MedAkinator(binary_df)

    try:
        run_cli(model, max_questions=args.max_questions, confidence_stop=args.confidence_stop,
                margin_stop=args.margin_stop, min_questions_for_stop=args.min_questions)
    except KeyboardInterrupt:
        print("\n\nSesi dihentikan (Ctrl+C).")


if __name__ == "__main__":
    main()