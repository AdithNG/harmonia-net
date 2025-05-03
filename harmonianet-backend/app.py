from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import torch
import librosa
import numpy as np
import psutil

from model import GenreCNN
from utils import genre_to_idx, idx_to_genre

app = FastAPI()

origins = [
    "https://harmonia-net.vercel.app",
    "https://harmonia-net.vercel.app/predict",
    "http://localhost:3000"             
]

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model once
model = GenreCNN(num_classes=len(genre_to_idx))
model.load_state_dict(torch.load('harmonianet.pth', map_location=torch.device('cpu')))
model.eval()

def extract_mel_from_bytes(audio_bytes, sr=22050, n_mels=128, fixed_len=1280):
    y, _ = librosa.load(audio_bytes, sr=sr, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < fixed_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, fixed_len - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :fixed_len]

    return torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("Starting prediction...", flush=True)
        audio_bytes = BytesIO(await file.read())

        print("Memory after file read (MB):", psutil.Process().memory_info().rss / 1024**2, flush=True)

        input_tensor = extract_mel_from_bytes(audio_bytes)

        print("Memory after preprocessing (MB):", psutil.Process().memory_info().rss / 1024**2, flush=True)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]
            result = {idx_to_genre[i]: float(p) for i, p in enumerate(probs)}

        print("Memory after prediction (MB):", psutil.Process().memory_info().rss / 1024**2, flush=True)
        print("Finished prediction.", flush=True)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "HarmoniaNet is live"}

@app.head("/predict")
async def predict_head():
    return JSONResponse(content=None, status_code=200)
