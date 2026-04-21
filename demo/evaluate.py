# --- evaluate.py ---
import pandas as pd
import numpy as np
import joblib
import json
import warnings
from supabase import create_client, Client
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI
# =============================================================================
DATASET_UJI_PATH = 'dataset_uji_sebenarnya.csv'
KOLOM_JAWABAN    = 'isFraud'
MODEL_ID_DI_DB   = 1

# --- KONEKSI SUPABASE ---
SUPABASE_URL = "https://foyygomirrulwokboexh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZveXlnb21pcnJ1bHdva2JvZXhoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMwOTU1MzIsImV4cCI6MjA3ODY3MTUzMn0.gShRgFJcANjQYl11Z32lZ6sHJE_qaO-_d_MIAQ7u2Mg"

# =============================================================================
# FITUR MODEL
# =============================================================================
MODEL_FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'Hour', 'Day', 'IsNight', 'IsBusinessHours',
    'amount_log', 'amount_sqrt',
    'balance_change_orig', 'balance_change_orig_pct', 'sender_depleted',
    'balance_change_dest', 'balance_change_dest_pct', 'receiver_new_account',
    'expected_balance_orig', 'balance_inconsistent',
    # Fitur Agregasi Historis
    'tx_count_24h', 'tx_count_7d',
    'avg_amount_30d', 'max_amount_30d', 'std_amount_30d',
    'is_amount_unusual', 'is_new_high_amount', 'amount_vs_avg_ratio',
    'usual_hour_customer', 'is_unusual_hour',
    'type_encoded'
]

# =============================================================================
# KONEKSI SUPABASE
# =============================================================================
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Koneksi Supabase Berhasil.")
except Exception as e:
    print(f"❌ Koneksi Gagal: {e}")
    exit()

# =============================================================================
# LOAD THRESHOLD DARI FILE JSON)
# =============================================================================
try:
    with open('threshold_config.json', 'r') as f:
        config = json.load(f)
        FINAL_THRESHOLD = config['optimal_threshold']
    print(f"✓ Threshold dimuat dari file: {FINAL_THRESHOLD}")
except FileNotFoundError:
    print("    File 'threshold_config.json' tidak ditemukan!")
    print("    Pastikan file ini ada di direktori yang sama dengan evaluate.py.")
    print("    File ini dihasilkan otomatis saat notebook training selesai dijalankan.")
    exit()  # Hentikan proses — jangan lanjut dengan threshold yang salah
except KeyError:
    print("    Key 'optimal_threshold' tidak ditemukan di threshold_config.json!")
    print("    Periksa format file JSON dari notebook Anda.")
    exit()
except Exception as e:
    print(f"Error membaca threshold: {e}")
    exit()

# =============================================================================
# LOAD MODEL & SCALER
# =============================================================================
try:
    model  = joblib.load('isolation_forest_baseline.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✓ Model & Scaler berhasil dimuat.")
except FileNotFoundError as e:
    print(f"File tidak ditemukan: {e}")
    exit()
except Exception as e:
    print(f"Gagal memuat model/scaler: {e}")
    exit()

# =============================================================================
# FUNGSI UTAMA
# =============================================================================
def main():
    print(f"\n{'='*60}")
    print(f"  ISOFRAUD - Evaluasi Model")
    print(f"  Threshold  : {FINAL_THRESHOLD}")
    print(f"  Dataset    : {DATASET_UJI_PATH}")
    print(f"  Jumlah Fitur: {len(MODEL_FEATURES)}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Baca Data Uji
    # ------------------------------------------------------------------
    try:
        df_test = pd.read_csv(DATASET_UJI_PATH)
        print(f"\n✓ Dataset dimuat: {len(df_test)} baris, {len(df_test.columns)} kolom")
    except FileNotFoundError:
        print(f"❌ File '{DATASET_UJI_PATH}' tidak ditemukan!")
        return
    except Exception as e:
        print(f"❌ Error membaca dataset: {e}")
        return

    # ------------------------------------------------------------------
    # 2. Validasi Kolom
    # ------------------------------------------------------------------
    if KOLOM_JAWABAN not in df_test.columns:
        print(f"❌ Kolom label '{KOLOM_JAWABAN}' tidak ada di dataset!")
        return

    kolom_hilang = [f for f in MODEL_FEATURES if f not in df_test.columns]
    if kolom_hilang:
        print(f"❌ Kolom fitur berikut tidak ditemukan di dataset:")
        for k in kolom_hilang:
            print(f"   - {k}")
        print("\n   Pastikan dataset_uji_sebenarnya.csv sudah melalui feature engineering")
        print("   yang SAMA dengan yang dilakukan di notebook training.")
        return

    # ------------------------------------------------------------------
    # 3. Pisahkan Fitur dan Label
    # ------------------------------------------------------------------
    y_true = df_test[KOLOM_JAWABAN]
    X_test = df_test[MODEL_FEATURES]

    fraud_count  = y_true.sum()
    normal_count = len(y_true) - fraud_count
    print(f"\n📊 Distribusi Label:")
    print(f"   Normal (0): {normal_count:,}  |  Fraud (1): {fraud_count:,}")

    # ------------------------------------------------------------------
    # 4. Scaling
    # ------------------------------------------------------------------
    print("\n⚙️  Melakukan Scaling...")
    try:
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        print(f"❌ Error saat scaling: {e}")
        return

    # ------------------------------------------------------------------
    # 5. Prediksi dengan Threshold yang Benar
    # ------------------------------------------------------------------
    print("🧠 Melakukan Prediksi...")
    scores         = model.decision_function(X_test_scaled)
    anomaly_scores = -scores  # Balik tanda: makin tinggi = makin anomali
    y_pred         = (anomaly_scores >= FINAL_THRESHOLD).astype(int)

    # ------------------------------------------------------------------
    # 6. Hitung Semua Metrik
    # ------------------------------------------------------------------
    print("📈 Menghitung Metrik Evaluasi...")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_true, anomaly_scores)
    pr_auc    = average_precision_score(y_true, anomaly_scores)
    mcc       = matthews_corrcoef(y_true, y_pred)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_val     = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_val     = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n{'='*60}")
    print(f"  HASIL EVALUASI")
    print(f"{'='*60}")
    print(f"  Accuracy   : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision  : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall     : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1-Score   : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"  PR-AUC     : {pr_auc:.4f}")
    print(f"  MCC        : {mcc:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  FPR        : {fpr_val:.4f}")
    print(f"  FNR        : {fnr_val:.4f}")
    print(f"{'='*60}")
    print(f"  TP: {int(tp):,}  |  FP: {int(fp):,}  |  TN: {int(tn):,}  |  FN: {int(fn):,}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 7. Konfirmasi Sebelum Simpan ke DB
    # ------------------------------------------------------------------
    konfirmasi = input("Simpan hasil ini ke Supabase? (y/n): ").strip().lower()
    if konfirmasi != 'y':
        print("ℹ️  Dibatalkan. Data tidak disimpan.")
        return

    evaluation_data = {
        'model_id'        : MODEL_ID_DI_DB,
        'evaluation_date' : datetime.now().isoformat(),
        'dataset_name'    : 'Test Set (Final)',
        'accuracy'        : float(accuracy),
        'precision'       : float(precision),
        'recall'          : float(recall),
        'f1_score'        : float(f1),
        'roc_auc'         : float(roc_auc),
        'pr_auc'          : float(pr_auc),
        'mcc'             : float(mcc),
        'specificity'     : float(specificity),
        'fpr'             : float(fpr_val),
        'fnr'             : float(fnr_val),
        'tp'              : int(tp),
        'tn'              : int(tn),
        'fp'              : int(fp),
        'fn'              : int(fn),
    }

    try:
        supabase.table('model_evaluation').insert(evaluation_data).execute()
        print("✅ BERHASIL! Data evaluasi terbaru telah disimpan ke Supabase.")
        print("   Silakan refresh halaman Admin → Evaluasi Model di website.")
    except Exception as e:
        print(f"❌ Gagal menyimpan ke database: {e}")


if __name__ == '__main__':
    main()