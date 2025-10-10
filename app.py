from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Helper to get absolute path for model files
def model_path(filename):
    return os.path.join(os.path.dirname(__file__), 'models', filename)

# Load model and encoders
model = joblib.load(model_path('xgb_model.pkl'))
scaler = joblib.load(model_path('scaler.pkl'))
le_genre = joblib.load(model_path('le_genre.pkl'))
le_time = joblib.load(model_path('le_time.pkl'))
le_day = joblib.load(model_path('le_day.pkl'))
le_sentiment = joblib.load(model_path('le_sentiment.pkl'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predictor')
def predictor():
    return render_template('index.html')

@app.route('/back')
def back():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        episode_length = float(request.form['episode_length'])
        guest_popularity = float(request.form['guest_popularity'])
        host_popularity = float(request.form['host_popularity'])
        num_ads = int(request.form['num_ads'])
        genre = request.form['genre']
        publication_day = request.form['publication_day']
        publication_time = request.form['publication_time']
        episode_sentiment = request.form['episode_sentiment']

        # Helper to safely transform categorical inputs, else raise error
        def safe_transform(le, val, field_name):
            if val not in le.classes_:
                raise ValueError(f"Invalid {field_name} value: '{val}'")
            return le.transform([val])[0]

        genre_encoded = safe_transform(le_genre, genre, "genre")
        day_encoded = safe_transform(le_day, publication_day, "publication_day")
        time_encoded = safe_transform(le_time, publication_time, "publication_time")
        sentiment_encoded = safe_transform(le_sentiment, episode_sentiment, "episode_sentiment")

        features = np.array([[episode_length, guest_popularity, host_popularity,
                              num_ads, day_encoded, time_encoded, genre_encoded, sentiment_encoded]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        return jsonify({'prediction': f'Estimated Listening Time: {prediction:.2f} minutes'})
    
    except Exception as e:
        return jsonify({'prediction': f'Error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)