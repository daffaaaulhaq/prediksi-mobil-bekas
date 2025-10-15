# app.py — Dashboard + Prediksi Harga Mobil Bekas
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime

import altair as alt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Paths & artefak
# -----------------------------
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", page_icon="🚗", layout="wide")

BASE = Path(__file__).resolve().parent
DIR_ART = BASE / "artefak"
DIR_TRAIN = BASE / "data_training"
DIR_TEST  = BASE / "data_testing"
META_PATH = DIR_ART / "meta.json"
MODEL_PATH = DIR_ART / "model_linreg.pkl"

assert META_PATH.exists(), f"meta.json tidak ditemukan: {META_PATH}"
meta = json.load(open(META_PATH, encoding="utf-8"))

TRAIN_CSV = DIR_TRAIN / meta.get("train_path", "train.csv")
assert TRAIN_CSV.exists(), f"train.csv tidak ditemukan: {TRAIN_CSV}"

num_cols = meta["num_cols"]
cat_cols = meta["cat_cols"]
TARGET = meta["target"]
use_log = meta.get("log_target", True)
THIS_YEAR = datetime.now().year

# -----------------------------
# Cache loading
# -----------------------------
# 👇 Tambahkan ini untuk PAKSA MENGHAPUS MODEL LAMA
if MODEL_PATH.exists():
    MODEL_PATH.unlink()  # Hapus file model lama agar retrain fresh
# Paksa hapus model lama supaya retrain sesuai meta.json sekarang
try:
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
except Exception:
    pass

@st.cache_resource(show_spinner=False)
def load_model():
    """Muat model; jika gagal (mismatch versi), retrain cepat dari TRAIN_CSV."""
    try:
        mdl = joblib.load(MODEL_PATH)
        retrained = False
    except Exception:
        train_df = pd.read_csv(TRAIN_CSV)
        X_train = train_df.drop(columns=[TARGET])
        y_train = train_df[TARGET]
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
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        mdl = pipe
        retrained = True
    return mdl, retrained

@st.cache_data(show_spinner=False)
def load_train():
    return pd.read_csv(TRAIN_CSV)

@st.cache_data(show_spinner=False)
def compute_mae_for_interval(_mdl, df_train):
    """Estimasi MAE memakai test.csv bila ada, jika tidak pakai holdout dari train."""
    test_path = DIR_TEST / meta.get("test_path", "test.csv")
    if test_path.exists():
        df_test = pd.read_csv(test_path)
        X = df_test.drop(columns=[TARGET])
        y = df_test[TARGET]
    else:
        df_train = df_train.sample(frac=1.0, random_state=42)
        n = max(int(0.2 * len(df_train)), 1)
        hold = df_train.iloc[:n].copy()
        X = hold.drop(columns=[TARGET])
        y = hold[TARGET]
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
    except Exception:
        return str(x)

def uniq(df, col):
    if col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).unique().tolist()
    vals = [v for v in vals if v and v.lower() != "nan"]
    return sorted(vals)

def prepare_input(
    tahun, jarak_km, cc_mid, merek, model_name, varian, bbm,
    transmisi, tipe_bodi, sistem_penggerak, warna
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
    }
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
# Sidebar Navigation
# -----------------------------
with st.sidebar:
    st.title("🚗 Navigasi")
    menu = st.radio(
        "Pilih Halaman",
        ["Dashboard", "Analisis Data", "Metode & Algoritma", "Prediksi Harga"],
        index=0
    )
    st.divider()
    st.caption(f"Model: Multiple Linear Regression (OLS)")
    st.caption(f"Log-target: **{'Ya' if use_log else 'Tidak'}**")
    st.caption(f"Retrain saat startup: **{'Ya' if retrained else 'Tidak'}**")
    st.caption(f"MAE perkiraan: **{rupiah(mae_est)}**")

# -----------------------------
# Page: Dashboard
# -----------------------------
def page_dashboard():
    st.title("🏠 Dashboard Proyek Prediksi Harga Mobil Bekas")
    st.markdown("### Masalah yang Kami Angkat")
    st.write(
        "- **Harga mobil bekas sulit diperkirakan** karena variabel yang beragam "
        "(merek, model, varian, tahun, jarak tempuh, bodi, transmisi, dsb).\n"
        "- Pelanggan kerap **kehilangan waktu** untuk membandingkan listing.\n"
        "- Penjual membutuhkan **alat bantu objektif** untuk memasang harga wajar."
    )

    st.markdown("### Solusi Kami")
    st.write(
        "- Membangun **model regresi** untuk _estimasi harga_ dari atribut kendaraan.\n"
        "- **Aplikasi web interaktif** (Streamlit) untuk input spesifikasi → keluar **perkiraan harga & kisaran (±MAE)**.\n"
        "- Menyediakan **dashboard analitik** agar pemangku kepentingan memahami sebaran data."
    )

    st.markdown("### Penjelasan Dataset")
    st.write(
        "- Sumber: dataset internal `mobilbekas.csv` (±14 ribu baris, 26 kolom).\n"
        "- Fitur numerik turunan: `Jarak_km`, `CC_mid` (midpoint kapasitas mesin), `Usia_mobil`.\n"
        "- Setelah _cleaning_: kolom lokasi/penjual dihapus agar fokus pada spesifikasi mobil."
    )

    st.markdown("### Metode (CRISP-DM)")
    st.write(
        "1. **Business Understanding** → masalah & tujuan bisnis\n"
        "2. **Data Understanding** → eksplorasi & kualitas data\n"
        "3. **Data Preparation** → pembersihan, transformasi, feature engineering\n"
        "4. **Modeling** → Multiple Linear Regression (OLS)\n"
        "5. **Evaluation** → RMSE/MAE/R²/MAPE\n"
        "6. **Deployment** → Aplikasi Streamlit"
    )

    st.markdown("### Wawasan Singkat (High-Level)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Jumlah Data Training", f"{len(train_df):,}")
    with c2:
        if TARGET in train_df.columns:
            st.metric("Harga Median", rupiah(train_df[TARGET].median()))
    with c3:
        st.metric("MAE (estimasi)", rupiah(mae_est))

    st.markdown("### Batasan")
    st.write(
        "- Model linear (OLS) **mengasumsikan linearitas** dan **independensi fitur**.\n"
        "- **Outlier** dan **kategori langka** bisa memengaruhi akurasi.\n"
        "- Prediksi **bukan harga final**, melainkan **estimasi** berdasarkan pola historis data."
    )

# -----------------------------
# Page: Analisis Data (EDA)
# -----------------------------
def page_eda():
    st.title("📊 Analisis Data")
    st.caption("Visualisasi dibuat dari data training agar konsisten dengan encoder.")

    # Ringkasan
    st.subheader("Ringkasan Statistik")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Jumlah baris**:", f"{len(train_df):,}")
    with c2:
        st.write("**Jumlah fitur numerik**:", len(num_cols))
    with c3:
        st.write("**Jumlah fitur kategori**:", len(cat_cols))

    st.divider()

    # Tambahan: ringkasan missing values
    st.subheader("Ringkasan Missing Values")
    miss = train_df.isna().sum().reset_index()
    miss.columns = ["Kolom", "JumlahMissing"]
    miss = miss[miss["JumlahMissing"]>0].sort_values("JumlahMissing", ascending=False)
    if len(miss) > 0:
        st.table(miss)
    else:
        st.write("Tidak ada missing values pada data training yang terdeteksi.")

    # Korelasi numerik (heatmap) — gunakan sample kecil untuk performa
    st.subheader("Korelasi Fitur Numerik")
    numdf = train_df[num_cols + [TARGET]] if TARGET in train_df.columns else train_df[num_cols]
    numdf = numdf.select_dtypes(include=[np.number])
    if numdf.shape[1] >= 2:
        corr = numdf.corr()
        corr_long = (
            corr.reset_index().melt(id_vars="index")
            .rename(columns={"index":"x", "variable":"y", "value":"corr"})
        )
        heat = alt.Chart(corr_long).mark_rect().encode(
            x=alt.X("x:N", sort=list(corr.columns)),
            y=alt.Y("y:N", sort=list(corr.columns)),
            color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1,1], scheme="redblue")),
            tooltip=[alt.Tooltip("x"), alt.Tooltip("y"), alt.Tooltip("corr:Q", format=".2f")]
        ).properties(height=350)
        st.altair_chart(heat, use_container_width=True)
    else:
        st.write("Tidak cukup fitur numerik untuk menampilkan korelasi.")

    # Distribusi harga (log vs linear)
    st.subheader("Distribusi Harga: Linear vs Log")
    if TARGET in train_df.columns:
        df_plot = train_df[[TARGET]].dropna()
        df_plot = df_plot.sample(n=min(len(df_plot), 20000), random_state=42)
        p1 = alt.Chart(df_plot).mark_bar().encode(
            alt.X(f"{TARGET}:Q", bin=alt.Bin(maxbins=80), title="Harga"),
            alt.Y('count()', title='Jumlah')
        ).properties(width=450, height=250)
        df_plot = df_plot.assign(log_price=np.log1p(df_plot[TARGET]))
        p2 = alt.Chart(df_plot).mark_bar().encode(
            alt.X("log_price:Q", bin=alt.Bin(maxbins=80), title="Log(Harga+1)"),
            alt.Y('count()', title='Jumlah')
        ).properties(width=450, height=250)
        st.altair_chart(alt.hconcat(p1, p2), use_container_width=True)

    # Scatter Harga vs Jarak (dengan garis regresi sederhana)
    st.subheader("Harga vs Jarak Tempuh")
    if "Jarak_km" in train_df.columns and TARGET in train_df.columns:
        df_sc = train_df[["Jarak_km", TARGET]].dropna()
        df_sc = df_sc[df_sc["Jarak_km"]>=0]
        df_sc = df_sc.sample(n=min(len(df_sc), 20000), random_state=42)
        scatter = alt.Chart(df_sc).mark_circle(size=30, opacity=0.3).encode(
            x=alt.X("Jarak_km:Q", title="Jarak (km)"),
            y=alt.Y(f"{TARGET}:Q", title="Harga", axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("Jarak_km:Q", title="Jarak (km)"), alt.Tooltip(f"{TARGET}:Q", title="Harga", format=",.0f")]
        ).properties(height=350)
        # tambah trendline (LOESS via transform_loess)
        loess = scatter.transform_loess('Jarak_km', TARGET, bandwidth=0.3).mark_line(color='red')
        st.altair_chart(scatter + loess, use_container_width=True)

    # Rata-rata Harga per Tahun
    st.subheader("Rata-rata Harga per Tahun")
    if "Tahun" in train_df.columns and TARGET in train_df.columns:
        avg_year = train_df.groupby("Tahun")[TARGET].mean().reset_index()
        line = alt.Chart(avg_year).mark_line(point=True).encode(
            x=alt.X("Tahun:O", title="Tahun"),
            y=alt.Y(f"{TARGET}:Q", title="Rata-rata Harga", axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("Tahun"), alt.Tooltip(f"{TARGET}:Q", format=",.0f")]
        ).properties(height=300)
        st.altair_chart(line, use_container_width=True)

    # Top Model / Varian
    st.subheader("Top Model & Varian berdasarkan Jumlah Data")
    if "Model" in train_df.columns:
        top_model = (
            train_df["Model"].value_counts()
            .reset_index(name="Jumlah")
            .rename(columns={"index":"Model"})
            .head(10)
        )
        # pastikan tipe string agar Altair tidak bingung
        top_model["Model"] = top_model["Model"].astype(str)
        bar_m = alt.Chart(top_model).mark_bar().encode(
            x=alt.X("Jumlah:Q", title="Jumlah"),
            y=alt.Y("Model:N", sort='-x', title="Model"),
            tooltip=[alt.Tooltip("Model:N", title="Model"), alt.Tooltip("Jumlah:Q", title="Jumlah")]
        ).properties(height=300)
        st.altair_chart(bar_m, use_container_width=True)


    # Distribusi Harga
    st.subheader("Distribusi Harga")
    if TARGET in train_df.columns:
        hist = alt.Chart(train_df).mark_bar().encode(
            alt.X(f"{TARGET}:Q", bin=alt.Bin(maxbins=50), title="Harga"),
            alt.Y('count()', title='Jumlah'),
            tooltip=[alt.Tooltip(f"{TARGET}:Q", title="Harga", format=",.0f"), alt.Tooltip('count()', title='n')]
        ).properties(height=280)
        st.altair_chart(hist, use_container_width=True)

    

    # Rata-rata harga per merek (Top 10 by count)
    st.subheader("Rata-rata Harga per Merek (Top 10 berdasarkan jumlah data)")
    if "Merek" in train_df.columns and TARGET in train_df.columns:
        vc = train_df["Merek"].value_counts()
        top10 = vc.index[:10]
        mean_price = (
            train_df[train_df["Merek"].isin(top10)]
            .groupby("Merek")[TARGET].mean().reset_index()
        )
        mean_price["Merek"] = mean_price["Merek"].astype(str)
        bar_mean = alt.Chart(mean_price).mark_bar().encode(
            x=alt.X(f"{TARGET}:Q", title="Rata-rata Harga", axis=alt.Axis(format=",.0f")),
            y=alt.Y("Merek:N", sort='-x', title="Merek"),
            tooltip=[alt.Tooltip("Merek:N", title="Merek"), alt.Tooltip(f"{TARGET}:Q", title="Rata-rata", format=",.0f")]
        ).properties(height=320)
        st.altair_chart(bar_mean, use_container_width=True)

    # Usia vs Harga (scatter)
    st.subheader("Usia Mobil vs Harga (Scatter)")
    if "Usia_mobil" in train_df.columns and TARGET in train_df.columns:
        scatter = alt.Chart(train_df).mark_circle(size=30, opacity=0.5).encode(
            x=alt.X("Usia_mobil:Q", title="Usia Mobil (tahun)"),
            y=alt.Y(f"{TARGET}:Q", title="Harga", axis=alt.Axis(format=",.0f")),
            tooltip=["Usia_mobil", alt.Tooltip(f"{TARGET}:Q", title="Harga", format=",.0f")]
        ).properties(height=320)
        st.altair_chart(scatter, use_container_width=True)

    # Boxplot Harga per Tipe Bodi
    st.subheader("Sebaran Harga per Tipe Bodi (Boxplot)")
    if "Tipe bodi" in train_df.columns and TARGET in train_df.columns:
        box = alt.Chart(train_df).mark_boxplot().encode(
            x=alt.X("Tipe bodi:N", title="Tipe Bodi"),
            y=alt.Y(f"{TARGET}:Q", title="Harga", axis=alt.Axis(format=",.0f")),
            tooltip=["Tipe bodi"]
        ).properties(height=340)
        st.altair_chart(box, use_container_width=True)
        

# -----------------------------
# Page: Metode & Algoritma
# -----------------------------
def page_method():
    st.title("🧠 Metode & Algoritma")
    st.markdown("### CRISP-DM")
    st.write(
        "- **Business Understanding**: Menentukan kebutuhan estimasi harga mobil bekas.\n"
        "- **Data Understanding**: Memahami struktur & kualitas data (missing value, outlier, kategori langka).\n"
        "- **Data Preparation**: Pembersihan, konversi satuan, feature-engineering (Jarak_km, CC_mid, Usia_mobil), drop kolom lokasi/penjual.\n"
        "- **Modeling**: Multiple Linear Regression (OLS) dengan One-Hot Encoding untuk fitur kategorikal, imputasi missing values.\n"
        "- **Evaluation**: RMSE, MAE, R², MAPE; bandingkan baseline/varian model.\n"
        "- **Deployment**: Aplikasi Streamlit + artefak model (.pkl)."
    )

    st.markdown("### Algoritma: Multiple Linear Regression (OLS)")
    st.write(
        "Kita memodelkan hubungan linear antara fitur-fitur X dan target harga Y.\n"
        "Encoding: **OneHotEncoder(handle_unknown='ignore', drop='first')** untuk menghindari dummy trap.\n"
        "Target di-log (opsional) agar error relatif stabil, lalu di-expm1 saat inferensi."
    )

    st.markdown("### Metrik Kinerja (Estimasi dari Holdout/Test)")
    st.write(f"- **MAE**: {rupiah(mae_est)} (kisaran harga ±MAE ditampilkan di halaman prediksi)")

# -----------------------------
# Page: Prediksi Harga
# -----------------------------
def page_predict():
    st.title("🔮 Prediksi Harga Mobil Bekas")
    st.caption("Isi atribut mobil (tanpa kolom lokasi/penjual). Model: Multiple Linear Regression (OLS).")

    col1, col2 = st.columns(2)

    with col1:
        tahun = st.number_input("Tahun", min_value=1990, max_value=THIS_YEAR, value=2018, step=1)
        jarak_km = st.number_input("Jarak tempuh (km)", min_value=0, max_value=2_000_000, value=30_000, step=1_000)
        cc_mid = st.number_input("Kapasitas mesin (cc)", min_value=500, max_value=7000, value=1500, step=50)

        merek_list = uniq(train_df, "Merek")
        merek = st.selectbox("Merek", options=merek_list or ["Toyota"], index=0)

        df_merek = train_df[train_df["Merek"].astype(str) == str(merek)]
        model_list = uniq(df_merek, "Model") or uniq(train_df, "Model")
        model_name = st.selectbox("Model", options=model_list or ["Avanza"], index=0)

        df_mm = df_merek[df_merek["Model"].astype(str) == str(model_name)]
        varian_list = uniq(df_mm, "Varian") or uniq(train_df, "Varian")
        varian = st.selectbox("Varian", options=varian_list or ["1.3 G"], index=0)

    with col2:
        bbm = st.selectbox("Tipe bahan bakar", options=uniq(train_df, "Tipe bahan bakar") or ["Bensin"])
        transmisi = st.selectbox("Transmisi", options=uniq(train_df, "Transmisi") or ["Automatic"])

        # Tipe Bodi otomatis berdasar Merek+Model+Varian → tampil sebagai field biasa (boleh diubah manual)
        df_bodi_auto = df_mm[df_mm["Varian"].astype(str) == str(varian)]
        bodi_list = df_bodi_auto["Tipe bodi"].dropna().unique().tolist()
        if len(bodi_list) == 0:
            default_bodi = ""
        elif len(bodi_list) == 1:
            default_bodi = bodi_list[0]
        else:
            default_bodi = df_bodi_auto["Tipe bodi"].mode()[0]
        tipe_bodi = st.text_input("Tipe bodi", value=default_bodi)

        drivetrain = st.selectbox("Sistem Penggerak", options=uniq(train_df, "Sistem Penggerak") or ["Front Wheel Drive (FWD)"])
        warna = st.selectbox("Warna", options=uniq(train_df, "Warna") or ["Hitam"])

    st.divider()
    c1, c2, _ = st.columns([1,1,6])
    with c1:
        predict_btn = st.button("Prediksi Harga", type="primary")
    with c2:
        if st.button("↺ Reset"):
            st.experimental_rerun()

    if predict_btn:
        X_in = prepare_input(
            tahun, jarak_km, cc_mid, merek, model_name, varian, bbm,
            transmisi, tipe_bodi, drivetrain, warna
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

# -----------------------------
# Router
# -----------------------------
if menu == "Dashboard":
    page_dashboard()
elif menu == "Analisis Data":
    page_eda()
elif menu == "Metode & Algoritma":
    page_method()
else:
    page_predict()
