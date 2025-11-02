from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sqlite3
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Create folder for uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained emotion detection model
model = load_model('face_emotionModel.h5')

# Emotion classes (in same order as training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        email TEXT,
                        image_path TEXT,
                        emotion_detected TEXT,
                        created_at TEXT
                      )''')
    conn.commit()
    conn.close()

init_db()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Emotion detection route
@app.route('/detect', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return "No image uploaded", 400

    name = request.form['name']
    email = request.form['email']
    file = request.files['image']

    if file.filename == '':
        return "No file selected", 400

    # Save image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image for model
    img = image.load_img(filepath, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict emotion
    prediction = model.predict(img_array)
    emotion_index = np.argmax(prediction)
    detected_emotion = emotion_labels[emotion_index]

    # Store data in SQLite database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name, email, image_path, emotion_detected, created_at) VALUES (?, ?, ?, ?, ?)',
                   (name, email, filepath, detected_emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    # Response message
    response = f"You look {detected_emotion.lower()}. "
    if detected_emotion.lower() in ['sad', 'angry', 'fear', 'disgust']:
        response += "Why are you sad?"
    else:
        response += "Glad to see you happy!"

    return f"""
    <h2>{response}</h2>
    <p><b>Name:</b> {name}</p>
    <p><b>Email:</b> {email}</p>
    <p><b>Detected Emotion:</b> {detected_emotion}</p>
    <img src='/{filepath}' width='200'>
    <br><br>
    <a href='/'>Go Back</a>
    """

if __name__ == '__main__':
    app.run(debug=True)
