import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv('diabetes_raw/diabetes.csv')

# =========================
# HAPUS DATA DUPLIKAT
# =========================
df = df.drop_duplicates()

# =========================
# MENANGANI NILAI 0
# =========================
columns_zero = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

for col in columns_zero:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# =========================
# STANDARDISASI
# =========================
scaler = StandardScaler()

X = df.drop('Outcome', axis=1)

X_scaled = scaler.fit_transform(X)

# =========================
# MEMBUAT DATAFRAME BARU
# =========================
scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

scaled_df['Outcome'] = df['Outcome']

# =========================
# SIMPAN DATA CLEAN
# =========================
scaled_df.to_csv(
    'diabetes_clean.csv',
    index=False
)

print("Preprocessing selesai!")

