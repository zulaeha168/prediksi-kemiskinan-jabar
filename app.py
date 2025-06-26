import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import altair as alt

# --------------------------------------------------
# Data Loading & Preprocessing
# --------------------------------------------------
data = pd.read_csv("Persentase Penduduk Miskin Menurut Kabupaten_Kota di Jawa Barat, 2024.csv")
data_clean = data.iloc[2:].reset_index(drop=True)
data_clean.columns = ["Kota", "Persentase"]
data_clean["Kota"] = data_clean["Kota"].str.strip()
data_clean["Persentase"] = data_clean["Persentase"].str.replace(",", ".").astype(float)

def categorize(value):
    if value <= 5:
        return "Rendah"
    elif value <= 10:
        return "Sedang"
    else:
        return "Tinggi"

data_clean["Kategori"] = data_clean["Persentase"].apply(categorize)

# --------------------------------------------------
# Model Training
# --------------------------------------------------
X = data_clean[["Persentase"]]
y = data_clean["Kategori"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
eval_report = classification_report(y_test, y_pred, output_dict=True)

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(page_title="Prediksi Kemiskinan Jawa Barat", layout="wide")

# --------------------------------------------------
# Header & Profil
# --------------------------------------------------
st.markdown("""
    <div style='background-color:#003366; padding:20px; border-radius:8px;'>
        <h1 style='color:white; text-align:center;'>🌍 Aplikasi Prediksi Kategori Kemiskinan Jawa Barat</h1>
        <p style='color:white; text-align:center; font-size:16px;'>
            Aplikasi ini memprediksi status tingkat kemiskinan daerah di Provinsi Jawa Barat berdasarkan persentase penduduk miskin.
            Menggunakan <b>Naive Bayes</b> dengan Gaussian Distribution, aplikasi ini membantu memetakan daerah <i>Rendah</i>, <i>Sedang</i>, atau <i>Tinggi</i> tingkat kemiskinan secara real-time.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Input dan Data Display
# --------------------------------------------------
col1, col2 = st.columns([1,2])

with col1:
    st.markdown("#### 🔍 Masukkan Persentase Kemiskinan")
    input_value = st.number_input("Persentase Penduduk Miskin (%)", min_value=0.0, max_value=100.0, value=7.46, step=0.1)
    predict_btn = st.button("🚀 Prediksi Kategori")
    if predict_btn:
        pred = model.predict(np.array([[input_value]]))[0]
        color = "#28a745" if pred == "Rendah" else ("#ffc107" if pred == "Sedang" else "#dc3545")
        st.markdown(f"""
            <div style='background-color:{color}; padding:20px; border-radius:8px; text-align:center; color:white; font-size:20px; font-weight:bold;'>
                ✅ Prediksi Status: {pred}
            </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("#### 📋 Daftar Daerah dan Kategorinya")
    st.dataframe(data_clean, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# Evaluasi Model
# --------------------------------------------------
st.markdown("""
    <div style='text-align:center;'>
        <h3>📈 Evaluasi Model</h3>
    </div>
""", unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
col3.metric("Akurasi", f"{eval_report['accuracy']:.2f}")
col4.metric("Presisi Rata-rata", f"{eval_report['weighted avg']['precision']:.2f}")
col5.metric("Recall Rata-rata", f"{eval_report['weighted avg']['recall']:.2f}")

st.markdown("""
    <div style='text-align:center; color:gray; font-size:13px;'>
        Model ini dapat dijadikan alat bantu analisis status kemiskinan daerah berbasis data BPS.
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Koefisiensi Model
# --------------------------------------------------
koef_data = {
    "Kategori": model.classes_,
    "Rata-rata (theta_)": model.theta_[:, 0],
    "Varian (sigma_)": model.var_[:, 0]  # <-- Diganti dari sigma_ menjadi var_
}
koef_df = pd.DataFrame(koef_data)

st.markdown("""
    <div style='text-align:center;' >
        <h3>📊 Koefisiensi Model</h3>
    </div>
""", unsafe_allow_html=True)

st.dataframe(koef_df, use_container_width=True)

st.markdown("""
    <div style='text-align:center; color:gray; font-size:13px;'>
        Nilai rata-rata dan variansi dari masing-masing kelas (Rendah, Sedang, dan Tinggi) digunakan oleh model Naive Bayes untuk menghitung nilai probabilitas prediksi status daerah.
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Visualisasi Status Daerah
# --------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align:center; font-size:18px; font-weight:bold;'>
        📊 Visualisasi Status Kemiskinan Daerah
    </div>
""", unsafe_allow_html=True)

# 1️⃣ Pie Chart
status_counts = data_clean["Kategori"].value_counts().reset_index()
status_counts.columns = ["Kategori", "Jumlah"]

pie_chart = alt.Chart(status_counts).mark_arc().encode(
    theta="Jumlah",
    color=alt.Color("Kategori:N", scale=alt.Scale(scheme="category10")),
    tooltip=["Kategori", "Jumlah"]
).properties(title="Distribusi Status Kemiskinan Daerah")

# 2️⃣ Bar Chart
bar_chart = alt.Chart(data_clean).mark_bar().encode(
    x=alt.X("Kota:N", sort="-y", title="Kota / Kabupaten"),
    y=alt.Y("Persentase:Q", title="Persentase Kemiskinan (%)"),
    color=alt.Color("Kategori:N", scale=alt.Scale(scheme="category10")),
    tooltip=["Kota", "Persentase", "Kategori"]
).properties(
    title="Nilai Persentase Kemiskinan Tiap Daerah",
    width=600,
    height=400
).configure_view(stroke=None)

# --------------------------------------------------
# Tampilan Visualisasi
# --------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.altair_chart(pie_chart, use_container_width=True)

with colB:
    st.altair_chart(bar_chart, use_container_width=True)

st.markdown("""
    <div style='text-align:center; color:gray; font-size:13px;'>
        Visualisasi ini memberikan gambaran pola tingkat kemiskinan daerah sehingga dapat digunakan sebagai bahan pertimbangan untuk pengambilan kebijakan daerah.
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Export Data
# --------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align:center; font-size:18px; font-weight:bold;'>
        📥 Export Hasil Prediksi
    </div>
""", unsafe_allow_html=True)

csv_data = data_clean.to_csv(index=False).encode("utf-8")
st.download_button(
    label="💾 Download Data Status Kemiskinan (CSV)",
    data=csv_data,
    file_name="status_kemiskinan_jawa_barat.csv",
    mime="text/csv"
)

st.markdown("""
    <div style='text-align:center; color:gray; font-size:13px;'>
        Unduh data lengkap untuk kebutuhan analisis lebih lanjut atau pelaporan.
    </div>
""", unsafe_allow_html=True)
