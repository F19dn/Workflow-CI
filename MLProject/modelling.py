
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Direktori dan dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "preprocessed_data.csv"

# Konfigurasi clustering
FEATURES = [
    "competitor_density_500m",
    "base_price_idr",
    "discount_voucher_pct",
    "daily_cups_sold",
    "customer_review_rating",
]

N_CLUSTERS = 5
RANDOM_STATE = 42

# Validasi dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

missing_columns = [
    column for column in FEATURES
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Kolom fitur tidak ditemukan: {missing_columns}"
    )

if len(df) < N_CLUSTERS:
    raise ValueError(
        f"Jumlah data harus minimal {N_CLUSTERS} baris."
    )

# Persiapan fitur
X = df[FEATURES].copy()

for column in FEATURES:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Mengisi missing values dengan median
X = X.fillna(X.median())

# Menghapus fitur yang seluruh nilainya kosong
X = X.dropna(axis=1, how="all")

# Menghapus fitur konstan
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.empty:
    raise ValueError(
        "Tidak ada fitur valid untuk proses clustering."
    )

if X.isna().any().any():
    raise ValueError(
        "Masih terdapat missing values pada fitur."
    )

# Konfigurasi MLflow Autolog
mlflow.set_experiment("Coffee Retail Market Clustering")
mlflow.sklearn.autolog()

# Pipeline clustering
model = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
    )),
])

# Training model
model.fit(X)

print("Model clustering selesai dilatih.")
print(f"Jumlah data: {len(X)}")
print(f"Jumlah fitur: {len(X.columns)}")
print(f"Jumlah cluster: {N_CLUSTERS}")
print("Parameter, metrik yang didukung, dan model dicatat melalui MLflow Autolog.")
