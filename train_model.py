import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\sophi\OneDrive\Documents\SIK\SEMESTER 4\DATA MINING\prediksi-suhu\underwater_temperature.csv")

# Normalisasi nama kolom
df.columns = df.columns.str.strip()
df.rename(columns=lambda x: x.replace('�', '°'), inplace=True)

# Drop nilai NaN pada suhu
df = df.dropna(subset=['Temp (°C)'])

# Pilih fitur dan target
X = df[['Latitude', 'Longitude', 'Depth']]
y = df['Temp (°C)']

# Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inisialisasi dan latih model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model ke file dengan nama yang sesuai Flask app
with open('model_rf.pkl', 'wb') as f:
    pickle.dump(model, f)


print("✅ Model berhasil dilatih dan disimpan sebagai 'model_rf.pkl'")
