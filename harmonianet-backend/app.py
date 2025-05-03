from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import librosa
import numpy as np
from io import BytesIO
import psutil

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
    print("Starting prediction...", flush = True)

    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    file = request.files['file']
    audio_bytes = BytesIO(file.read())

    try:
        print("Memory after file read (MB):", psutil.Process().memory_info().rss / 1024**2, flush = True)

        y, sr = librosa.load(audio_bytes, sr=22050, duration=30)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] < 1280:
            mel_db = np.pad(mel_db, ((0, 0), (0, 1280 - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :1280]

        input_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()

        print("Memory after preprocessing (MB):", psutil.Process().memory_info().rss / 1024**2, flush = True)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]
            result = {idx_to_genre[i]: float(p) for i, p in enumerate(probs)}

        print("Memory after prediction (MB):", psutil.Process().memory_info().rss / 1024**2, flush = True)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    print("Finished prediction.", flush = True)
    return jsonify(result)

@app.route('/', methods=['HEAD'])
def index():
    return '', 200

if __name__ == '__main__':
    # This block is unused when deploying with Gunicorn
    '''
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    '''
    pass
    
