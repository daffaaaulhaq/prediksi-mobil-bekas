import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime
import streamlit.components.v1 as components

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Paths & artefak
# -----------------------------
BASE = Path(__file__).resolve().parent
DIR_ART = BASE / "artefak"
DIR_TRAIN = BASE / "data_training"
META_PATH = DIR_ART / "meta.json"
MODEL_PATH = DIR_ART / "model_linreg.pkl"

assert META_PATH.exists(), f"meta.json tidak ditemukan: {META_PATH}"
meta = json.load(open(META_PATH))

TRAIN_CSV = DIR_TRAIN / meta.get("train_path", "train.csv")
assert TRAIN_CSV.exists(), f"train.csv tidak ditemukan: {TRAIN_CSV}"

num_cols = meta["num_cols"]
cat_cols = meta["cat_cols"]
TARGET = meta["target"]
use_log = meta.get("log_target", True)
THIS_YEAR = datetime.now().year

# Kita tidak tampilkan Penjual & Nama Bursa Mobil di UI.
# Jika "Nama Bursa Mobil" ada dalam cat_cols, kita akan isi NaN saat inferensi.
HIDE_COLS = {"Nama Bursa Mobil", "Penjual"}

# -----------------------------
# Cache loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        mdl = joblib.load(MODEL_PATH)
        retrained = False
    except Exception:
        # jika pickle mismatch → retrain dari train.csv
        train_df = pd.read_csv(TRAIN_CSV)
        X_train = train_df.drop(columns=[TARGET])
        y_train = train_df[TARGET]
        # bangun pipeline minimal sama dengan training
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh",  OneHotEncoder(handle_unknown="ignore", drop="first")),
        ])
        prep = ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ])
        linreg = LinearRegression()
        pipe = Pipeline([("prep", prep), ("model", linreg)])
        if use_log:
            pipe.fit(X_train, np.log1p(y_train))
        else:
            pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_PATH)
        mdl = pipe
        retrained = True
    return mdl, retrained

@st.cache_data(show_spinner=False)
def load_train():
    return pd.read_csv(TRAIN_CSV)

@st.cache_data(show_spinner=False)
def compute_mae_for_interval(_mdl, df_train):
    # pakai test.csv jika ada, kalau tidak pakai holdout sederhana dari train
    test_path = BASE / "data_testing" / "test.csv"
    if test_path.exists():
        df_test = pd.read_csv(test_path)
        X = df_test.drop(columns=[TARGET])
        y = df_test[TARGET]
    else:
        df_train = df_train.sample(frac=1.0, random_state=42)
        n = int(0.2 * len(df_train))
        df_test = df_train.iloc[:n].copy()
        X = df_test.drop(columns=[TARGET])
        y = df_test[TARGET]

    pred = _mdl.predict(X)
    if use_log:
        pred = np.expm1(pred)
    mae = mean_absolute_error(y, pred)
    return float(mae)


# -----------------------------
# Helpers
# -----------------------------
def rupiah(x):
    try:
        return "Rp {:,.0f}".format(float(x)).replace(",", ".")
    except:
        return str(x)

def uniq(df, col):
    vals = df[col].dropna().astype(str).unique().tolist()
    vals = [v for v in vals if v and v.lower() != "nan"]
    return sorted(vals)

def prepare_input(
    tahun, jarak_km, cc_mid, merek, model_name, varian, bbm,
    transmisi, tipe_bodi, sistem_penggerak, warna, kotakab, provinsi
):
    usia = THIS_YEAR - tahun if pd.notna(tahun) else np.nan
    row = {
        "Tahun": tahun,
        "Jarak_km": jarak_km,
        "CC_mid": cc_mid,
        "Usia_mobil": usia,
        "Merek": merek,
        "Model": model_name,
        "Varian": varian,
        "Tipe bahan bakar": bbm,
        "Transmisi": transmisi,
        "Tipe bodi": tipe_bodi,
        "Sistem Penggerak": sistem_penggerak,
        "Warna": warna,
        "KotaKab": kotakab,
        "Provinsi": provinsi,
        # Kolom yang disembunyikan di-NaN-kan agar kompatibel dengan pipeline
        "Tipe Penjual": np.nan if "Tipe Penjual" in cat_cols else None,
        "Nama Bursa Mobil": np.nan if "Nama Bursa Mobil" in cat_cols else None,
    }
    # pastikan semua kolom yang diminta model tersedia & berurutan
    needed = num_cols + cat_cols
    for c in needed:
        if c not in row:
            row[c] = np.nan
    return pd.DataFrame([row])[needed]

# -----------------------------
# Load resources
# -----------------------------
model, retrained = load_model()
train_df = load_train()
mae_est = compute_mae_for_interval(model, train_df)

# -----------------------------
# UI
# -----------------------------

# ====== THEME SWITCHER ======
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", page_icon="🚗", layout="wide")

st.title("Prediksi Harga Mobil Bekas — Web Interaktif")
st.caption("Isi atribut mobil (tanpa nama pemilik & showroom). Model: Multiple Linear Regression (OLS).")


# ====== FILTERED DROPDOWNS ======
# Input kiri-kanan
col1, col2 = st.columns(2)

with col1:
    tahun = st.number_input("Tahun", min_value=1990, max_value=THIS_YEAR, value=2018, step=1)
    jarak_km = st.number_input("Jarak tempuh (km)", min_value=0, max_value=2_000_000, value=30_000, step=1_000)
    cc_mid = st.number_input("Kapasitas mesin (cc)", min_value=500, max_value=7000, value=1500, step=50)

    # Merek
    merek_list = uniq(train_df, "Merek")
    merek = st.selectbox("Merek", options=merek_list or ["Toyota"], index=0)

    # Model tergantung merek
    df_merek = train_df[train_df["Merek"].astype(str) == str(merek)]
    model_list = uniq(df_merek, "Model") or uniq(train_df, "Model")
    model_name = st.selectbox("Model", options=model_list or ["Avanza"], index=0)

    # Varian tergantung merek+model
    df_mm = df_merek[df_merek["Model"].astype(str) == str(model_name)]
    varian_list = uniq(df_mm, "Varian") or uniq(train_df, "Varian")
    varian = st.selectbox("Varian", options=varian_list or ["1.3 G"], index=0)

with col2:
    bbm = st.selectbox("Tipe bahan bakar", options=uniq(train_df, "Tipe bahan bakar") or ["Bensin"])
    transmisi = st.selectbox("Transmisi", options=uniq(train_df, "Transmisi") or ["Automatic"])
    tipe_bodi = st.selectbox("Tipe bodi", options=uniq(train_df, "Tipe bodi") or ["MPV"])
    drivetrain = st.selectbox("Sistem Penggerak", options=uniq(train_df, "Sistem Penggerak") or ["Front Wheel Drive (FWD)"])
    warna = st.selectbox("Warna", options=uniq(train_df, "Warna") or ["Hitam"])
    kotakab = st.selectbox("Kota/Kab", options=uniq(train_df, "KotaKab") or ["Bandung Kota"])
    provinsi = st.selectbox("Provinsi", options=uniq(train_df, "Provinsi") or ["Jawa Barat"])

st.divider()
c1, c2, c3 = st.columns([1,1,6])
with c1:
    predict_btn = st.button("Prediksi Harga", type="primary")
with c2:
    reset = st.button("↺ Reset")

if reset:
    st.experimental_rerun()

if predict_btn:
    X_in = prepare_input(
        tahun, jarak_km, cc_mid, merek, model_name, varian, bbm,
        transmisi, tipe_bodi, drivetrain, warna, kotakab, provinsi
    )
    y_hat = model.predict(X_in)
    if use_log:
        y_hat = np.expm1(y_hat)
    price = float(y_hat[0])
    lo = max(price - mae_est, 0)
    hi = price + mae_est

    st.subheader("Hasil Prediksi")
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            f"**Perkiraan Harga:**<br>"
            f"<span style='font-size:22px;color:#4CAF50;'>{rupiah(price)}</span>",
            unsafe_allow_html=True
        )
    with colB:
        st.markdown(
            f"**Kisaran (± MAE):**<br>"
            f"<span style='font-size:22px;color:#2196F3;'>{rupiah(lo)} – {rupiah(hi)}</span>",
            unsafe_allow_html=True
        )
    st.caption("Kisaran berdasarkan MAE pada data uji/holdout. Bukan garansi harga.")
else:
    st.info("Silakan isi atribut mobil dan tekan **Prediksi Harga** untuk melihat hasil.")


