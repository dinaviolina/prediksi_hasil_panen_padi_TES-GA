import streamlit as st
import pandas as pd
import pickle
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error

# Load model and scalers
with open('model-tes.pkl', 'rb') as file:
    model_tes = pickle.load(file)
def load_model():
    try:
        with open('model-tes-ga.pkl', 'rb') as f:
            model_tes_ga = pickle.load(f)
        return model_tes_ga
    except FileNotFoundError:
        st.error("Model TES-GA tidak ditemukan. Silakan latih model terlebih dahulu.")
        return None
def load_scalers():
    with open('scaler_X.pkl', 'rb') as scaler_x_file:
        scaler_X = pickle.load(scaler_x_file)
    with open('scaler_y.pkl', 'rb') as scaler_y_file:
        scaler_y = pickle.load(scaler_y_file)
    return scaler_X, scaler_y

st.markdown("<h1 style='text-align: center;'>Sistem Prediksi Hasil Panen Padi di Madura</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Menggunakan Metode TES-GA</h3>", unsafe_allow_html=True)
st.image("padi.png", use_container_width=True, caption="Oleh: Dina Violina")

# sidebar
st.sidebar.image("padi.png", use_container_width=True)
st.sidebar.header("Menu")
menu_selection = st.sidebar.selectbox("Pilih menu:", ["Upload Data", "Preprocessing", "Modelling", "Pilih Model", "Prediksi"])

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'normalized_data' not in st.session_state:
    st.session_state.normalized_data = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'alpha' not in st.session_state:
    st.session_state.alpha = 10000.0
if 'beta' not in st.session_state:
    st.session_state.beta = 0.001
if 'gamma' not in st.session_state:
    st.session_state.gamma = 1.0
if 'seasonal_periods' not in st.session_state:
    st.session_state.seasonal_periods = 12
if 'population_size' not in st.session_state:
    st.session_state.population_size = 100
if 'model_tes_ga' not in st.session_state:
    model_tes_ga = load_model()
    scaler_X, scaler_y = load_scalers()
    st.session_state.model_tes_ga = model_tes_ga
    st.session_state.scaler_X = scaler_X
    st.session_state.scaler_y = scaler_y


# def train_tes_with_ga(X_train, y_train):
#     population_size = st.session_state.population_size
#     num_generations = st.session_state.num_generations

#     best_alpha = np.random.uniform(0, 1.0)
#     best_beta = np.random.uniform(0, 1.0)
#     best_gamma = np.random.uniform(0, 1.0)

#     # Melatih model TES dengan parameter terbaik yang ditemukan oleh GA
#     ga_model = ExponentialSmoothing(seasonal_periods=seasonal_periods, alpha=best_alpha, beta=best_beta, gamma=best_gamma)
#     ga_model.fit(X_train, y_train)
#     return ga_model

#1. Display content based on the selected menu
if menu_selection == "Upload Data":
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
    if uploaded_file:
        try:
            # Read the uploaded file and store it in session state
            st.session_state.data = pd.read_csv(uploaded_file)
            # Remove commas and convert to float
            for column in st.session_state.data.select_dtypes(include=[object]).columns:
                st.session_state.data[column] = (
                    st.session_state.data[column]
                    .apply(lambda x: float(str(x).replace(",", "")))
                )
            st.success("Data berhasil diunggah.")
            st.dataframe(st.session_state.data)
            st.write("Statistik data:")
            st.write(st.session_state.data.describe())
            st.session_state.selected_features = st.multiselect("Pilih fitur yang digunakan untuk model:", st.session_state.data.columns.tolist())
            if st.session_state.selected_features:
                st.write("Fitur yang dipilih:", st.session_state.selected_features)
        except Exception as e:
            st.error(f"Error saat memproses file: {e}")
# 2.
elif menu_selection == "Preprocessing":
    st.subheader("Preprocessing")
    if st.session_state.data is not None and st.session_state.selected_features:
        st.write("Data yang diunggah dengan fitur terpilih:")
        st.dataframe(st.session_state.data[st.session_state.selected_features])
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        st.session_state.data[st.session_state.selected_features] = imputer.fit_transform(st.session_state.data[st.session_state.selected_features])
        # Normalize selected features after removing commas
        for column in st.session_state.selected_features:
            if column in st.session_state.data.columns:
                st.session_state.data[column] = st.session_state.data[column].astype(str).str.replace(',', '').astype(float)
        # Remove outliers from specified columns
        for column in ['luas panen', 'produktivitas', 'hasil']:
            if column in st.session_state.data.columns:
                Q1 = st.session_state.data[column].quantile(0.25)
                Q3 = st.session_state.data[column].quantile(0.75)
                IQR = Q3 - Q1
                outlier_condition = (st.session_state.data[column] >= (Q1 - 1.5 * IQR)) & (st.session_state.data[column] <= (Q3 + 1.5 * IQR))
                st.session_state.data = st.session_state.data[outlier_condition]
        # Save cleaned data
        st.session_state.cleaned_data = st.session_state.data[st.session_state.selected_features]
        
        st.write("Data setelah penghapusan outlier:")
        st.dataframe(st.session_state.cleaned_data)

        # Normalize selected features
        scaler_X = MinMaxScaler()
        st.session_state.normalized_data = pd.DataFrame(scaler_X.fit_transform(st.session_state.cleaned_data), columns=st.session_state.selected_features)

        st.write("Hasil Data Setelah Normalisasi:")
        st.dataframe(st.session_state.normalized_data)
    else:
        st.warning("Data belum diunggah atau fitur belum dipilih, silakan unggah data dan pilih fitur terlebih dahulu.")
# 3. Modelling
elif menu_selection == "Modelling":
    st.subheader("Modelling")
    
    # Ensure data is normalized and selected
    if st.session_state.normalized_data is not None and st.session_state.data is not None:
        st.write("Melakukan split data menjadi training dan testing.")
        # Choose data split ratio
        split_option = st.selectbox("Pilih rasio pembagian data training dan testing:", ["50:50", "60:40", "70:30", "80:20", "90:10"])
        test_size = {
            "50:50": 0.5,
            "60:40": 0.4,
            "70:30": 0.3,
            "80:20": 0.2,
            "90:10": 0.1
        }[split_option]

        # Determine features and target
        X = st.session_state.normalized_data[['luas panen', 'produktivitas']]
        y = st.session_state.data['hasil']

        # Ensure X and y are consistent in size
        if len(X) != len(y):
            st.warning(f"Jumlah sampel tidak konsisten: X memiliki {len(X)} sampel, tetapi y memiliki {len(y)} sampel.")
        else:
            # Check for missing values before splitting
            if X.isnull().any().any() or y.isnull().any():
                st.warning("Terdapat nilai hilang dalam data. Silakan periksa dan isi nilai hilang sebelum melanjutkan.")
            else:
                try:
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                    # Save the split results to session state
                    st.session_state.X_train = X_train
                    st.session_state.y_train = y_train
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    st.write(f"Data dibagi dengan rasio {split_option}.")
                    st.write(f"Jumlah Data Training: {len(X_train)}")
                    st.write(f"Jumlah Data Testing: {len(X_test)}")
                    
                except Exception as e:
                    st.error(f"Error saat membagi data: {e}")
    else:
        st.warning("Silakan unggah data dan lakukan preprocessing terlebih dahulu.")
# 4. Pilih Model
elif menu_selection == "Pilih Model":
    st.header("Pilih Model")
    if st.session_state.data is not None:
        selected_model = st.radio(
            "Pilih Model:", ["TES", "TES-GA"]
        )
        st.session_state.selected_model = selected_model
        # TES
        if selected_model == "TES":
            if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
                st.warning("Silakan lakukan pemisahan data terlebih dahulu (pada bagian Modelling).")
            else:
                seasonal_periods = st.number_input(
                    "Masukkan periode musiman (seasonal periods):", 
                    min_value=2, 
                    value=int(st.session_state.get('seasonal_periods', 12)),  # Menggunakan integer untuk value default
                    step=1
                )
                st.session_state.seasonal_periods = seasonal_periods  # Menyimpan periode musiman ke session_state
                st.write(f"Periode musiman: **{seasonal_periods}**"
                )
                alpha = st.number_input(
                    "Masukkan nilai alpha:", 
                    min_value=0.0, 
                    value=float(st.session_state.get('alpha', 0.0))  # Pastikan ini selalu float
                )
                st.session_state.alpha = alpha  # Menyimpan nilai alpha
                beta = st.number_input(
                    "Masukkan nilai beta:", 
                    min_value=0.0, 
                    value=float(st.session_state.get('beta', 0.0))  # Pastikan ini selalu float
                )
                st.session_state.beta = beta  # Menyimpan nilai beta
                gamma = st.number_input(
                    "Masukkan nilai gamma:", 
                    min_value=0.0, 
                    value=float(st.session_state.get('gamma', 0.0))  # Pastikan ini selalu float
                )
                st.session_state.gamma = gamma  # Menyimpan nilai gamma

                # Menampilkan nilai-nilai yang disimpan di session_state
                st.write("Nilai alpha:", st.session_state.alpha)
                st.write("Nilai beta:", st.session_state.beta)
                st.write("Nilai gamma:", st.session_state.gamma)

                # Pastikan data pelatihan (y_train dan X_train) sudah ada
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train

                # Melatih model TES
                model = ExponentialSmoothing(
                    y_train, 
                    seasonal_periods=seasonal_periods, 
                    trend='add', 
                    seasonal='add', 
                    initialization_method='estimated',
                )
                model_fit = model.fit(
                )
                y_pred = model_fit.forecast(len(st.session_state.y_test))  # Menggunakan y_test dari session_state
                mape = mean_absolute_percentage_error(st.session_state.y_test, y_pred)
                st.write(f"MAPE : {mape:.2f}%")

        elif selected_model == "TES-GA":
            st.write("Model TES-GA")
            # Cek apakah model sudah ada
            model_tes_ga = load_model()
            if model_tes_ga is not None:
                st.write("Model TES-GA berhasil dimuat!")
                # Input parameter untuk TES dan GA
                population_size = st.number_input("Masukkan ukuran populasi (GA) :", min_value=1, value=10, step=1)
                num_generations = st.number_input("Masukkan jumlah generasi (GA) :", min_value=1, value=10, step=1)
                probabilitas_crossover = st.number_input("Masukkan probabilitas crossover (GA) :", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
                probabilitas_mutasi = st.number_input("Masukkan probabilitas mutasi (GA) :", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
                
                # Simpan parameter GA di session state
                st.session_state.population_size = population_size
                st.session_state.num_generations = num_generations
                st.session_state.probabilitas_crossover = probabilitas_crossover
                st.session_state.probabilitas_mutasi = probabilitas_mutasi
                
                # Input parameter TES (Exponential Smoothing)
                seasonal_periods = st.number_input("Masukkan periode musiman TES:", min_value=1, value=12, step=1)
                alpha = st.number_input("Masukkan alpha TES:", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
                beta = st.number_input("Masukkan beta TES:", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
                gamma = st.number_input("Masukkan gamma TES:", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
                
                # Simpan parameter TES di session state
                st.session_state.seasonal_periods = seasonal_periods
                st.session_state.alpha = alpha
                st.session_state.beta = beta
                st.session_state.gamma = gamma

                # Prediksi menggunakan model yang sudah dimuat dan parameter baru
                if 'X_train' in st.session_state and 'y_train' in st.session_state:
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    
                    # Membuat dan melatih model Exponential Smoothing dengan parameter yang diberikan
                    model = ExponentialSmoothing(
                        y_train, 
                        seasonal_periods=seasonal_periods, 
                        trend='add', 
                        seasonal='add', 
                        initialization_method='estimated', 
                    )
                    model_fit = model.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)

                    if 'y_test' in st.session_state:
                        y_test = st.session_state.y_test
                        y_pred = model_fit.forecast(len(y_test))
                        mape = mean_absolute_percentage_error(y_test, y_pred)
                        st.write(f"MAPE : {mape:.2f}%")
                    else:
                        st.warning("Data `y_test` tidak ditemukan. Pastikan data untuk testing telah diunggah.")
                else:
                    st.warning("Data training belum tersedia. Silahkan unggah data terlebih dahulu.")

# Prediksi berdasarkan input pengguna
elif menu_selection == "Prediksi":
    st.subheader("Prediksi")
    luaspanen = st.number_input("Masukkan Luas Panen (hektar):", min_value=0.0, format="%.2f")
    produktivitas = st.number_input("Masukkan produktivitas (ton)", min_value=0.0, format="%.2f")

    if st.button("Hitung hasil prediksi"):
        input_data = np.array([[luaspanen, produktivitas]])
        model_tes_ga = st.session_state.model_tes_ga
        scaler_X = st.session_state.scaler_X
        scaler_y = st.session_state.scaler_y
        input_data_normalized = scaler_X.transform(input_data)
        input_data_df = pd.DataFrame(input_data_normalized, columns=['luas panen', 'produktivitas'])
        input_data_df.index = pd.date_range(start='2021-01-01', periods=len(input_data_df), freq='M')
        try:
            hasil_panen_scaled =model_tes_ga.forecast(steps=len(input_data_df))
            hasil_panen = scaler_y.inverse_transform(hasil_panen_scaled.reshape(-1, 1))
            nilai_hasil = hasil_panen[0][0]

            st.success(f'Hasil prediksi: {nilai_hasil:.2f} ton')
        except Exception as e:
            st.error(f"Error saat melakukan prediksi: {e}")

