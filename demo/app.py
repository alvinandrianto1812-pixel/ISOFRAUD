# --- 1. IMPORT LIBRARY ---
import os
import pandas as pd
import numpy as np
import joblib  
import json
import warnings
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    session, jsonify, send_file
)
from supabase import create_client, Client
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from io import BytesIO

# --- 2. KONFIGURASI APLIKASI ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci-rahasia-anda-yang-sangat-aman-12345'

# --- 3. KONEKSI SUPABASE ---
SUPABASE_URL = "https://foyygomirrulwokboexh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZveXlnb21pcnJ1bHdva2JvZXhoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMwOTU1MzIsImV4cCI6MjA3ODY3MTUzMn0.gShRgFJcANjQYl11Z32lZ6sHJE_qaO-_d_MIAQ7u2Mg"

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Koneksi ke Supabase BERHASIL.")
except Exception as e:
    print(f"✗ Koneksi ke Supabase GAGAL: {e}")

# --- 4. LOAD MODEL DAN "OTAK" ---
MODEL_PATH     = 'isolation_forest_baseline.pkl'
SCALER_PATH    = 'scaler.pkl'
ENCODER_PATH   = 'label_encoders.pkl'
THRESHOLD_PATH = 'threshold_config.json'

def load_model():
    try:
        model    = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODER_PATH)
        with open(THRESHOLD_PATH, 'r') as f:
            threshold_config = json.load(f)
            threshold = threshold_config['optimal_threshold']
        print("✓ Model, Scaler, Encoders, dan Threshold berhasil dimuat.")
        return model, scaler, encoders, threshold
    except FileNotFoundError as e:
        print(f"✗ Error: File tidak ditemukan - {e}")
        return None, None, None, -0.0402
    except Exception as e:
        print(f"✗ Error saat memuat model: {e}")
        return None, None, None, -0.0402

if_model, scaler, label_encoders, FINAL_THRESHOLD = load_model()

# --- 5. DEFINISI FITUR MODEL ---
MODEL_FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'Hour', 'Day', 'IsNight', 'IsBusinessHours',
    'amount_log', 'amount_sqrt',
    'balance_change_orig', 'balance_change_orig_pct', 'sender_depleted',
    'balance_change_dest', 'balance_change_dest_pct', 'receiver_new_account',
    'expected_balance_orig', 'balance_inconsistent',
    'tx_count_24h', 'tx_count_7d',
    'avg_amount_30d', 'max_amount_30d', 'std_amount_30d',
    'is_amount_unusual', 'is_new_high_amount', 'amount_vs_avg_ratio',
    'usual_hour_customer', 'is_unusual_hour',
    'type_encoded'
]

# --- 6. FUNGSI FEATURE ENGINEERING ---

def create_time_features(df):
    df_out = df.copy()
    df_out['Hour'] = df_out['step'] % 24
    df_out['Day']  = df_out['step'] // 24
    df_out['IsNight']         = ((df_out['Hour'] >= 22) | (df_out['Hour'] <= 6)).astype(int)
    df_out['IsBusinessHours'] = ((df_out['Hour'] >= 9)  & (df_out['Hour'] <= 17)).astype(int)
    return df_out

def create_amount_features(df):
    df_out = df.copy()
    df_out['amount_log']  = np.log1p(df_out['amount'])
    df_out['amount_sqrt'] = np.sqrt(df_out['amount'])
    return df_out

def create_balance_features(df):
    df_out = df.copy()
    df_out['balance_change_orig']     = df_out['newbalanceOrig'] - df_out['oldbalanceOrg']
    df_out['balance_change_orig_pct'] = df_out['balance_change_orig'] / (df_out['oldbalanceOrg'] + 1)
    df_out['sender_depleted']         = (df_out['newbalanceOrig'] == 0).astype(int)
    df_out['balance_change_dest']     = df_out['newbalanceDest'] - df_out['oldbalanceDest']
    df_out['balance_change_dest_pct'] = df_out['balance_change_dest'] / (df_out['oldbalanceDest'] + 1)
    df_out['receiver_new_account']    = (df_out['oldbalanceDest'] == 0).astype(int)
    df_out['expected_balance_orig'] = df_out['oldbalanceOrg'] - df_out['amount']
    df_out['balance_inconsistent']  = (
        (df_out['expected_balance_orig'] != df_out['newbalanceOrig']) &
        (df_out['type'].isin(['CASH_OUT', 'TRANSFER']))
    ).astype(int)
    return df_out

def create_historical_aggregation_features(df):
    df_out = df.copy()
    df_out = df_out.sort_values(['nameOrig', 'step']).reset_index(drop=True)

    df_out['tx_count_24h'] = df_out.groupby('nameOrig')['step'].transform(
        lambda x: x.rolling(window=24, min_periods=1).count()
    ) - 1

    df_out['tx_count_7d'] = df_out.groupby('nameOrig')['step'].transform(
        lambda x: x.rolling(window=7 * 24, min_periods=1).count()
    ) - 1

    df_out['avg_amount_30d'] = df_out.groupby('nameOrig')['amount'].transform(
        lambda x: x.shift(1).rolling(window=30 * 24, min_periods=1).mean()
    )
    df_out['max_amount_30d'] = df_out.groupby('nameOrig')['amount'].transform(
        lambda x: x.shift(1).rolling(window=30 * 24, min_periods=1).max()
    )
    df_out['std_amount_30d'] = df_out.groupby('nameOrig')['amount'].transform(
        lambda x: x.shift(1).rolling(window=30 * 24, min_periods=1).std()
    )

    median_amount = df_out['amount'].median()
    df_out['avg_amount_30d'].fillna(median_amount, inplace=True)
    df_out['max_amount_30d'].fillna(median_amount, inplace=True)
    df_out['std_amount_30d'].fillna(0, inplace=True)

    df_out['is_amount_unusual']   = (df_out['amount'] > (3 * df_out['avg_amount_30d'])).astype(int)
    df_out['is_new_high_amount']  = (df_out['amount'] > df_out['max_amount_30d']).astype(int)
    df_out['amount_vs_avg_ratio'] = df_out['amount'] / (df_out['avg_amount_30d'] + 1)

    df_out['usual_hour_customer'] = df_out.groupby('nameOrig')['Hour'].transform(
        lambda x: x.expanding().apply(
            lambda y: y.iloc[:-1].mode()[0]
            if len(y) > 1 and len(y.iloc[:-1].mode()) > 0
            else -1,
            raw=False
        )
    )
    df_out['usual_hour_customer'].fillna(-1, inplace=True)

    df_out['is_unusual_hour'] = (
        (df_out['usual_hour_customer'] != -1) &
        (df_out['Hour'] != df_out['usual_hour_customer'])
    ).astype(int)

    return df_out


def encode_categorical(df, encoders):
    df_out  = df.copy()
    encoder = encoders.get('type')
    if encoder:
        df_out['type_encoded'] = df_out['type'].apply(
            lambda x: encoder.transform([str(x)])[0]
            if str(x) in encoder.classes_
            else -1
        )
    else:
        print("⚠️  Warning: Encoder untuk 'type' tidak ditemukan.")
        df_out['type_encoded'] = -1
    return df_out


def run_prediction_pipeline(df_input, model, scaler, encoders):
    print(f"\n{'='*70}")
    print(f"🔧 RUNNING PREDICTION PIPELINE")
    print(f"{'='*70}")
    print(f"Input shape: {df_input.shape}")
    print(f"Input columns: {df_input.columns.tolist()}\n")
    
    df_features = df_input.copy()
    
    # Step 1: Time features
    df_features = create_time_features(df_features)
    print(f"✅ After time features: {df_features.shape[1]} columns")
    
    # Step 2: Amount features
    df_features = create_amount_features(df_features)
    print(f"✅ After amount features: {df_features.shape[1]} columns")
    
    # Step 3: Balance features
    df_features = create_balance_features(df_features)
    print(f"✅ After balance features: {df_features.shape[1]} columns")
    
    # Step 4: Historical aggregation
    df_features = create_historical_aggregation_features(df_features)
    print(f"✅ After historical aggregation: {df_features.shape[1]} columns")
    
    # Step 5: Encode categorical
    df_features = encode_categorical(df_features, encoders)
    print(f"✅ After encoding: {df_features.shape[1]} columns")

    # Validate MODEL_FEATURES exist
    for col in MODEL_FEATURES:
        if col not in df_features.columns:
            print(f"⚠️  Warning: Fitur '{col}' tidak ditemukan, diisi 0.")
            df_features[col] = 0

    # Extract features for model
    df_final    = df_features[MODEL_FEATURES]
    data_scaled = scaler.transform(df_final)

    # Predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        scores = model.decision_function(data_scaled)

    anomaly_scores = -scores
    predictions    = (anomaly_scores >= FINAL_THRESHOLD).astype(int)

    # Add predictions to full dataframe (WITH all engineered features)
    df_results = df_features.copy()
    df_results['Fraud_Prediksi'] = predictions
    df_results['Anomaly_Score']  = anomaly_scores
    
    # Validate output
    print(f"\n📦 FINAL OUTPUT")
    print(f"{'='*70}")
    print(f"Total columns: {df_results.shape[1]}")
    
    # Check critical display columns
    display_check = ['Hour', 'IsNight', 'is_amount_unusual', 'is_new_high_amount', 'sender_depleted']
    missing = [c for c in display_check if c not in df_results.columns]
    
    if missing:
        print(f"❌ MISSING DISPLAY COLUMNS: {missing}")
    else:
        print(f"✅ All display columns present!")
    
    print(f"\nAll columns:")
    for i, col in enumerate(df_results.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"{'='*70}\n")

    total_fraud = int(predictions.sum())
    summary = {
        'total_data':     int(len(df_results)),
        'fraud_detected': total_fraud,
        'non_fraud':      int(len(df_results)) - total_fraud,
    }
    return df_results, summary

@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if request.method == 'POST':
        email            = request.form.get('email', '').strip()
        password         = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not email or not password:
            flash('Email dan password tidak boleh kosong.', 'error')
            return redirect(url_for('signup_page'))

        if password != confirm_password:
            flash('Password dan Konfirmasi Password tidak cocok.', 'error')
            return redirect(url_for('signup_page'))

        if len(password) < 6:
            flash('Password minimal 6 karakter.', 'error')
            return redirect(url_for('signup_page'))

        try:
            existing = supabase.table('user').select('id').eq('email', email).execute()
            if existing.data:
                flash('Email sudah terdaftar. Silakan Sign In.', 'error')
                return redirect(url_for('signup_page'))

            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            supabase.table('user').insert({
                'email':         email,
                'password_hash': hashed_password,
                'role':          'user'
            }).execute()

            flash('Akun berhasil dibuat! Silakan Sign In.', 'success')
            return redirect(url_for('signin_page'))

        except Exception as e:
            flash(f'Terjadi error saat mendaftar: {str(e)}', 'error')
            print(f"Error Sign Up: {e}")
            return redirect(url_for('signup_page'))

    return render_template('signup.html')


@app.route('/', methods=['GET', 'POST'])
@app.route('/signin', methods=['GET', 'POST'])
def signin_page():
    if 'user_id' in session:
        if session.get('user_role') == 'admin':
            return redirect(url_for('admin_dashboard_page'))
        return redirect(url_for('dashboard_page'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        try:
            response = supabase.table('user').select(
                'id, email, password_hash, role'
            ).eq('email', email).execute()

            if not response.data:
                flash('Email atau password salah.', 'error')
                return redirect(url_for('signin_page'))

            user = response.data[0]

            if check_password_hash(user['password_hash'], password):
                session['user_id']    = user['id']
                session['user_email'] = user['email']
                session['user_role']  = user['role']

                if user['role'] == 'admin':
                    return redirect(url_for('admin_dashboard_page'))
                return redirect(url_for('dashboard_page'))
            else:
                flash('Email atau password salah.', 'error')
                return redirect(url_for('signin_page'))

        except Exception as e:
            flash(f'Terjadi error: {str(e)}', 'error')
            print(f"Error Sign In: {e}")
            return redirect(url_for('signin_page'))

    return render_template('signin.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah logout.', 'success')
    return redirect(url_for('signin_page'))


# --- 8. RUTE USER ---

@app.route('/dashboard')
def dashboard_page():
    if 'user_id' not in session or session.get('user_role') != 'user':
        return redirect(url_for('signin_page'))
    return render_template('user/dashboarduser.html', user_email=session['user_email'])


@app.route('/upload', methods=['POST'])
def upload_dataset():
    """
    SELALU mengembalikan JSON — dipanggil via fetch() di dashboarduser.html.
    """
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Sesi habis. Silakan login kembali.'}), 401

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file yang diupload.'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'Tidak ada file yang dipilih.'})

    if not file.filename.lower().endswith('.csv'):
        return jsonify({'success': False, 'error': 'Hanya file CSV yang diperbolehkan.'})

    dataset_id = None

    try:
        df = pd.read_csv(file)

        required_columns = [
            'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
            'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'
        ]
        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            return jsonify({
                'success': False,
                'error': f'Kolom yang diperlukan tidak ada: {", ".join(missing_cols)}'
            })

        if if_model is None or scaler is None or label_encoders is None:
            return jsonify({
                'success': False,
                'error': 'Model belum dimuat di server. Hubungi administrator.'
            })

        # Upload original CSV ke Supabase Storage
        safe_filename  = secure_filename(file.filename)
        timestamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
        storage_path   = f"user_uploads/{session['user_id']}_{timestamp}_{safe_filename}"

        file.seek(0)
        file_bytes = file.read()
        supabase.storage.from_('datasets').upload(
            path=storage_path,
            file=file_bytes,
            file_options={'content-type': 'text/csv'}
        )

        # Insert metadata ke tabel dataset
        dataset_meta = {
            'user_id':           session['user_id'],
            'original_filename': safe_filename,
            'storage_path':      storage_path,
            'upload_at':         datetime.now().isoformat(),
            'row_count':         len(df),
            'status':            'Processing'
        }
        db_resp    = supabase.table('dataset').insert(dataset_meta).execute()
        dataset_id = db_resp.data[0]['dataset_id']

        # =====================================================================
        # PERBAIKAN: Run prediction pipeline
        # =====================================================================
        df_results, summary = run_prediction_pipeline(df, if_model, scaler, label_encoders)

        # =====================================================================
        # PERBAIKAN: Simpan hasil LENGKAP dengan SEMUA fitur engineered
        # =====================================================================
        import io
        result_csv_buffer = io.BytesIO()
        
        # SIMPAN df_results YANG SUDAH BERISI SEMUA FITUR ENGINEERED
        df_results.to_csv(result_csv_buffer, index=False)
        result_csv_buffer.seek(0)
        result_bytes = result_csv_buffer.read()

        # Upload hasil CSV ke Storage
        result_path = f"results/{dataset_id}_result.csv"
        supabase.storage.from_('datasets').upload(
            path=result_path,
            file=result_bytes,
            file_options={'content-type': 'text/csv'}
        )

        # Simpan summary ke detection_report
        import json as json_lib
        supabase.table('detection_report').insert({
            'dataset_id':    dataset_id,
            'result_path':   result_path,
            'summary_stats': json_lib.dumps(summary),
            'created_at':    datetime.now().isoformat()
        }).execute()

        # Update status dataset
        supabase.table('dataset').update({
            'status':       'Completed',
            'fraud_count':  summary['fraud_detected'],
            'processed_at': datetime.now().isoformat()
        }).eq('dataset_id', dataset_id).execute()

        return jsonify({
            'success':    True,
            'dataset_id': dataset_id,
            'summary':    summary,
            'message':    (
                f'Analisis selesai! '
                f'Ditemukan {summary["fraud_detected"]} transaksi fraud '
                f'dari {summary["total_data"]} total transaksi.'
            )
        })

    except Exception as e:
        import traceback
        print(f"Error detail upload: {e}")
        traceback.print_exc()

        if dataset_id:
            try:
                supabase.table('dataset').update(
                    {'status': 'Error'}
                ).eq('dataset_id', dataset_id).execute()
            except Exception:
                pass

        return jsonify({
            'success': False,
            'error':   f'Error saat memproses file: {str(e)}'
        })

@app.route('/history')
def history_page():
    if 'user_id' not in session or session.get('user_role') != 'user':
        return redirect(url_for('signin_page'))

    try:
        datasets = supabase.table('dataset').select('*').eq(
            'user_id', session['user_id']
        ).order('upload_at', desc=True).execute()

        return render_template(
            'user/history.html',
            datasets=datasets.data,
            user_email=session['user_email']
        )
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('dashboard_page'))


@app.route('/report/<int:dataset_id>')
def view_report(dataset_id):
    if 'user_id' not in session:
        return redirect(url_for('signin_page'))

    try:
        # 1. Ambil info dataset
        dataset_res = supabase.table('dataset').select('*').eq('dataset_id', dataset_id).execute()
        if not dataset_res.data:
            flash('Dataset tidak ditemukan.', 'error')
            return redirect(url_for('history_page'))

        dataset_info = dataset_res.data[0]

        if str(dataset_info.get('user_id')) != str(session['user_id']) \
                and session.get('user_role') != 'admin':
            flash('Akses ditolak.', 'error')
            return redirect(url_for('history_page'))

        # 2. Ambil report record
        report_res = (
            supabase.table('detection_report')
            .select('*')
            .eq('dataset_id', dataset_id)
            .limit(1)
            .execute()
        )

        if not report_res.data:
            flash('Laporan belum tersedia.', 'error')
            return redirect(url_for('history_page'))

        report_info   = report_res.data[0]
        summary_stats = report_info.get('summary_stats', {})

        if isinstance(summary_stats, str):
            import json as _json
            summary_stats = _json.loads(summary_stats)

        summary = {
            'total_data':     summary_stats.get('total_data', 0),
            'fraud_detected': summary_stats.get('fraud_detected', 0),
            'non_fraud':      summary_stats.get('non_fraud', 0),
        }

        # 3. Download CSV hasil dan ambil fraud data
        result_path = report_info.get('result_path', '')
        fraud_data  = []

        if result_path:
            try:
                import io
                csv_bytes  = supabase.storage.from_('datasets').download(result_path)
                df_result  = pd.read_csv(io.BytesIO(csv_bytes))
                
                print(f"\n{'='*60}")
                print(f"📋 DEBUG INFO - Dataset ID: {dataset_id}")
                print(f"{'='*60}")
                print(f"Total rows in CSV: {len(df_result)}")
                print(f"Columns: {df_result.columns.tolist()}")
                
                if 'Fraud_Prediksi' in df_result.columns:
                    print(f"\n🔍 Fraud_Prediksi column info:")
                    print(f"  Data type: {df_result['Fraud_Prediksi'].dtype}")
                    print(f"  Unique values: {df_result['Fraud_Prediksi'].unique()}")
                    print(f"  Value counts:")
                    print(df_result['Fraud_Prediksi'].value_counts())
                
                # =====================================================
                # PERBAIKAN: Deteksi nama kolom anomaly score
                # =====================================================
                anomaly_col = None
                if 'Anomaly Score' in df_result.columns:
                    anomaly_col = 'Anomaly Score'
                elif 'Anomaly_Score' in df_result.columns:
                    anomaly_col = 'Anomaly_Score'
                else:
                    print("⚠️  Warning: Anomaly score column not found!")
                
                # Filter fraud
                if 'Fraud_Prediksi' in df_result.columns:
                    df_result['Fraud_Prediksi'] = pd.to_numeric(df_result['Fraud_Prediksi'], errors='coerce')
                    df_fraud = df_result[df_result['Fraud_Prediksi'] >= 0.5].copy()
                    
                    print(f"\n✅ Fraud records found: {len(df_fraud)}")
                    
                    if len(df_fraud) > 0:
                        # Sort by anomaly score jika ada
                        if anomaly_col:
                            df_fraud = df_fraud.sort_values(anomaly_col, ascending=False).head(50)
                        else:
                            df_fraud = df_fraud.head(50)
                        
                        # =====================================================
                        # PERBAIKAN: Rename kolom untuk konsistensi dengan template
                        # =====================================================
                        if 'Anomaly_Score' in df_fraud.columns and 'Anomaly Score' not in df_fraud.columns:
                            df_fraud.rename(columns={'Anomaly_Score': 'Anomaly Score'}, inplace=True)
                        
                        # Select columns to display
                        display_columns = [
                            'step', 'type', 'amount',
                            'oldbalanceOrg', 'newbalanceOrig',
                            'Hour', 'IsNight', 
                            'is_amount_unusual', 'is_new_high_amount', 'sender_depleted',
                            'Anomaly Score'
                        ]

                        # Only include columns that exist
                        display_columns = [col for col in display_columns if col in df_fraud.columns]
                        
                        print(f"📊 Display columns: {display_columns}")
                        print(f"{'='*60}\n")

                        fraud_data = df_fraud[display_columns].to_dict('records')
                    else:
                        print("⚠️  No fraud data to display")
                else:
                    print("❌ Cannot filter - Fraud_Prediksi column missing")
                
            except Exception as ex:
                print(f'❌ Error reading result CSV: {ex}')
                import traceback
                traceback.print_exc()

        return render_template(
            'user/report.html',
            dataset=dataset_info,
            fraud_data=fraud_data,
            summary=summary,
            user_email=session['user_email']
        )

    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        print(f"❌ Error di /report: {e}")
        import traceback
        traceback.print_exc()
        return redirect(url_for('history_page'))


@app.route('/download/<int:dataset_id>')
def download_report(dataset_id):
    if 'user_id' not in session:
        return redirect(url_for('signin_page'))

    try:
        # Ambil result_path dari detection_report
        report_res = (
            supabase.table('detection_report')
            .select('result_path')
            .eq('dataset_id', dataset_id)
            .limit(1)
            .execute()
        )

        if not report_res.data or not report_res.data[0].get('result_path'):
            flash('Tidak ada data untuk didownload.', 'error')
            return redirect(url_for('history_page'))

        result_path = report_res.data[0]['result_path']

        # Download file CSV hasil dari Storage
        import io
        csv_bytes  = supabase.storage.from_('datasets').download(result_path)
        csv_buffer = BytesIO(csv_bytes)
        csv_buffer.seek(0)

        dataset = supabase.table('dataset').select('original_filename').eq(
            'dataset_id', dataset_id
        ).execute()
        original_filename = (
            dataset.data[0]['original_filename'] if dataset.data else 'report'
        )
        filename = f"fraud_detection_{original_filename}"

        return send_file(
            csv_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )

    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        print(f"Error download: {e}")
        return redirect(url_for('history_page'))


# --- 9. RUTE ADMIN ---

@app.route('/admin/dashboard')
def admin_dashboard_page():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return redirect(url_for('signin_page'))
    return render_template('admin/dashboardadmin.html', admin_email=session['user_email'])


@app.route('/admin/users')
def admin_users():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return redirect(url_for('signin_page'))

    try:
        users_res = supabase.table('user').select(
            'id, email, created_at'
        ).neq('role', 'admin').execute()

        user_list = []
        for user in users_res.data:
            count_res = supabase.table('dataset').select(
                'dataset_id', count='exact'
            ).eq('user_id', user['id']).execute()
            user_list.append({
                'id':           user['id'],
                'email':        user['email'],
                'created_at':   user['created_at'],
                'upload_count': count_res.count or 0,
            })

        return render_template(
            'admin/manageusers.html',
            users=user_list,
            admin_email=session['user_email']
        )
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        print(f"Error di /admin/users: {e}")
        return redirect(url_for('admin_dashboard_page'))


@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        print(f"--- Memulai penghapusan User ID: {user_id} ---")
        datasets_res = supabase.table('dataset').select('dataset_id').eq('user_id', user_id).execute()
        dataset_ids  = [d['dataset_id'] for d in datasets_res.data]

        if dataset_ids:
            supabase.table('detection_report').delete().in_('dataset_id', dataset_ids).execute()
            supabase.table('dataset').delete().in_('dataset_id', dataset_ids).execute()

        supabase.table('user').delete().eq('id', user_id).execute()
        print(f"SUKSES: User {user_id} berhasil dihapus.")
        return jsonify({'success': True, 'message': 'User dan semua datanya berhasil dihapus'})

    except Exception as e:
        print(f"Error di /admin/users/delete: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/admin/evaluation')
def admin_evaluation():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return redirect(url_for('signin_page'))

    try:
        evaluation = supabase.table('model_evaluation').select('*').order(
            'evaluation_date', desc=True
        ).limit(1).execute()

        if not evaluation.data:
            flash('Belum ada data evaluasi model di database.', 'info')
            return render_template(
                'admin/evaluation.html',
                evaluation=None,
                admin_email=session['user_email']
            )

        return render_template(
            'admin/evaluation.html',
            evaluation=evaluation.data[0],
            admin_email=session['user_email']
        )
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        print(f"Error di /admin/evaluation: {e}")
        return redirect(url_for('admin_dashboard_page'))


@app.route('/admin/evaluation/download')
def admin_download_metrics():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return redirect(url_for('signin_page'))

    try:
        evaluation = supabase.table('model_evaluation').select('*').order(
            'evaluation_date', desc=True
        ).limit(1).execute()

        if not evaluation.data:
            flash('Belum ada data evaluasi untuk didownload.', 'error')
            return redirect(url_for('admin_evaluation'))

        metrics_df = pd.DataFrame([evaluation.data[0]])
        csv_buffer = BytesIO()
        metrics_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename  = f"Model_Evaluation_Metrics_{timestamp}.csv"

        return send_file(
            csv_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )

    except Exception as e:
        flash(f'Gagal mendownload metrik: {str(e)}', 'error')
        print(f"Error download metrics: {e}")
        return redirect(url_for('admin_evaluation'))


@app.route('/admin/dataset_preview')
def admin_dataset_preview():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return redirect(url_for('signin_page'))

    try:
        df_before = pd.read_csv('demo_transactions.csv')

        if not label_encoders:
            raise Exception("Label Encoders tidak dimuat.")

        df_after = df_before.copy()
        df_after = create_time_features(df_after)
        df_after = create_amount_features(df_after)
        df_after = create_balance_features(df_after)
        df_after = create_historical_aggregation_features(df_after)
        df_after = encode_categorical(df_after, label_encoders)

        data_before_html = df_before.head(15).to_html(classes='dataframe', border=0)

        preview_cols = MODEL_FEATURES + ['step', 'type']
        preview_cols = [c for c in preview_cols if c in df_after.columns]
        data_after_html = df_after[preview_cols].head(15).to_html(classes='dataframe', border=0)

        return render_template(
            'admin/dataset_preview.html',
            admin_email=session['user_email'],
            data_before_html=data_before_html,
            data_after_html=data_after_html
        )

    except FileNotFoundError:
        flash('Error: file demo_transactions.csv tidak ditemukan.', 'error')
        return redirect(url_for('admin_dashboard_page'))
    except Exception as e:
        flash(f'Error saat memproses data: {str(e)}', 'error')
        print(f"Error di /admin/dataset_preview: {e}")
        return redirect(url_for('admin_dashboard_page'))


# --- 10. JALANKAN APLIKASI ---
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 FRAUD DETECTION SYSTEM - ISOFRAUD")
    print("=" * 80)
    print(f"Model Status:    {'✓ Loaded' if if_model else '✗ Not Loaded'}")
    print(f"Scaler Status:   {'✓ Loaded' if scaler else '✗ Not Loaded'}")
    print(f"Encoders Status: {'✓ Loaded' if label_encoders else '✗ Not Loaded'}")
    print(f"Threshold:       {FINAL_THRESHOLD}")
    print(f"Total Features:  {len(MODEL_FEATURES)}")
    print("=" * 80 + "\n")

    app.run(debug=True)