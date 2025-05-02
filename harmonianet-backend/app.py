from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchaudio
import librosa
import numpy as np
import os

# Load model architecture
from model import GenreCNN  # You’ll define this exactly like in Colab
from utils import genre_to_idx, idx_to_genre  # You can store these dicts in utils.py

app = Flask(__name__)
CORS(app)

# Load model
model = GenreCNN(num_classes=len(genre_to_idx))
model.load_state_dict(torch.load('harmonianet.pth', map_location=torch.device('cpu')))
model.eval()

def extract_mel(path, sr=22050, n_mels=128, fixed_len=1280):
    y, _ = librosa.load(path, sr=sr, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or trim
    if mel_db.shape[1] < fixed_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, fixed_len - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :fixed_len]

    return torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()  # (1, 1, n_mels, time)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    file = request.files['file']
    temp_path = "temp.wav"
    file.save(temp_path)

    try:
        input_tensor = extract_mel(temp_path)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]
            result = {idx_to_genre[i]: float(p) for i, p in enumerate(probs)}
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(temp_path)

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

