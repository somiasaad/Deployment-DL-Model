import numpy as np
from flask import Flask, request, render_template, jsonify
import librosa
from tensorflow.keras.models import Sequential, model_from_json
import pickle

# Flask app initialization
flask_app = Flask(__name__)

# Load the model
json_file = open('CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load the model weights
loaded_model.load_weights("best_model1_weights.h5")

# Load the scaler
with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

# Define feature extraction functions
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse1 = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse1)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])

    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.array(res)
    result = np.reshape(result, newshape=(1, 2376))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)

    return final_result

emotions = {1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fear', 7: 'Disgust', 8: 'Surprise'}

def make_prediction(path1):
    res = get_predict_feat(path1)
    predictions = loaded_model.predict(res)
    y_pred = np.argmax(predictions)
    return y_pred

# Flask routes
@flask_app.route("/")
def home():
    return render_template("chat.html")

@flask_app.route("/predict_route", methods=["POST"])
def predict_route():
    if 'audio_file' not in request.files:
        return "No audio file uploaded", 400
    audio_file = request.files['audio_file']
    print("=" * 50)
    print("audio file: ",audio_file)
    print("=" * 50)
    if not audio_file:
        audio_file = "audio.mp3"

    prediction = make_prediction(audio_file)

    return jsonify({"emotion": emotions[prediction]})

    # return render_template("chat.html", prediction_text=f"Type Of Emotion is {emotions[prediction]}".title())

# Main function
if __name__ == "__main__":
    flask_app.run(debug=False, host="0.0.0.0")