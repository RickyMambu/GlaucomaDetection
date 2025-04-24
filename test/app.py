# Import library yang diperlukan
from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
import json
import requests
import markdown
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Mendapatkan API key dari file .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY tidak ditemukan di file .env")
else:
    print(f"Loaded API Key: {api_key[:5]}...")  # Menampilkan sebagian API key untuk keamanan

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Format file yang diizinkan untuk diunggah
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Fungsi untuk memeriksa apakah file yang diunggah memiliki ekstensi yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Memuat model TensorFlow untuk prediksi
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_history.keras')
model = tf.keras.models.load_model(model_path)

# Daftar nama kelas untuk prediksi
class_names = ['Normal', 'Glaukoma']

# Fungsi untuk mendapatkan akurasi model dari file training_history.json
def get_model_accuracy():
    history_path = os.path.join(os.path.dirname(__file__), 'model', 'training_history.json')
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
            val_accuracy = history.get('val_accuracy', [])
            if val_accuracy:
                return f"{int(max(val_accuracy) * 100)}%"
    except Exception as e:
        print(f"Error loading training history: {e}")
    return "N/A"

# Fungsi untuk memprediksi gambar yang diunggah
def predict_image(filepath):
    # Memuat gambar dan mengubahnya menjadi array
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Melakukan prediksi menggunakan model
    pred = model.predict(img_array)
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = pred[0][predicted_class]
    predicted_label = class_names[predicted_class]

    # Mengembalikan label prediksi dan tingkat keyakinan
    return predicted_label, confidence

# Fungsi untuk mendapatkan deskripsi dari Gemini API berdasarkan hasil prediksi
def get_llm_description(predicted_label, confidence):
    try:
        # URL endpoint API Gemini
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        # Header untuk permintaan API
        headers = {
            "Content-Type": "application/json"
        }

        # Membuat prompt untuk dikirim ke Gemini API
        prompt = (
            f"Hasil prediksi menunjukkan kondisi: '{predicted_label}' dengan tingkat keyakinan {confidence*100:.2f}%. "
            f"Berikan analisis mendalam tentang kondisi ini, termasuk:\n\n"
            f"1. Penjelasan umum tentang kondisi '{predicted_label}'.\n"
            f"2. Risiko atau dampak jika kondisi ini tidak ditangani.\n"
            f"3. Saran awal untuk langkah-langkah yang dapat diambil mengobati atau mencegah Glaukoma jika.\n"
            f"4. Kapan pengguna harus segera memeriksakan diri ke dokter mata.\n\n"
            f"Selalu tekankan bahwa ini hanya informasi awal, konsultasi dokter sangat disarankan untuk dilakukan"
        )

        # Payload untuk permintaan API
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        # Mengirim permintaan POST ke API
        response = requests.post(api_url, json=payload, headers=headers)
        print("FULL JSON RESPONSE:", response.json())  # Debugging respons API

        # Memproses respons API
        if response.status_code == 200:
            candidates = response.json().get("candidates", [{}])
            description = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            html_description = markdown.markdown(description)  # Mengubah teks menjadi HTML
            return html_description
        else:
            return f"<p>Error: {response.status_code} - {response.text}</p>"
    except Exception as e:
        print(f"Error connecting to Gemini API: {e}")
        return "<p>Terjadi kesalahan saat meminta penjelasan dari Gemini.</p>"

# Route untuk halaman utama
@app.route('/')
def index():
    model_accuracy = get_model_accuracy()
    return render_template('index.html', model_accuracy=model_accuracy)

# Route untuk prediksi gambar yang diunggah
@app.route('/predict', methods=['POST'])
def predict():
    # Memeriksa apakah file diunggah
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Memeriksa apakah file memiliki format yang diizinkan
    if not allowed_file(file.filename):
        return "Invalid file type. Only images are allowed.", 400

    # Menyimpan file yang diunggah
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Melakukan prediksi dan mendapatkan deskripsi
    predicted_label, confidence = predict_image(filepath)
    image_url = url_for('static', filename=f'uploads/{filename}')
    model_accuracy = get_model_accuracy()
    description = get_llm_description(predicted_label, confidence)

    # Mengirim data ke template HTML
    return render_template('index.html',
                           prediction=predicted_label,
                           confidence=f"{confidence*100:.2f}%",
                           image_path=image_url,
                           model_accuracy=model_accuracy,
                           description=description)

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
