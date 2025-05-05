from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model
with open("model_rf.pkl", "rb") as f:
    model = pickle.load(f)

# Folder untuk simpan gambar
STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', temperature=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    depth = float(request.form['depth'])

    # Prediksi suhu
    features = np.array([[latitude, longitude, depth]])
    temperature = model.predict(features)[0]
    temperature = round(temperature, 2)

    # Buat visualisasi
    depth_range = np.linspace(0, 100, 100)
    simulated_data = np.array([[latitude, longitude, d] for d in depth_range])
    predicted_temp = model.predict(simulated_data)

    # Plot
    plt.figure(figsize=(5, 3))
    plt.plot(depth_range, predicted_temp, color='blue')
    plt.axvline(x=depth, color='red', linestyle='--', label='Kedalaman input')
    plt.xlabel('Kedalaman (m)')
    plt.ylabel('Suhu (°C)')
    plt.title('Prediksi Suhu vs Kedalaman')
    plt.legend()
    plt.tight_layout()

    plot_filename = os.path.join(STATIC_FOLDER, 'plot.png')
    plt.savefig(plot_filename)
    plt.close()

    return render_template('index.html',
                           latitude=latitude,
                           longitude=longitude,
                           depth=depth,
                           temperature=temperature,
                           plot_url='/static/plot.png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
