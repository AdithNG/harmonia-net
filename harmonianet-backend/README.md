# 🎛️ harmonianet-backend

This is the backend service for HarmoniaNet - a music genre classification app built using FastAPI and PyTorch. It handles audio file uploads, converts them into mel spectrograms, and uses a pretrained CNN model to predict the music genre.

---

## 🚀 Features

- 🎵 Accepts `.mp3` uploads through a `/predict` API endpoint
- 🔊 Converts audio to mel spectrograms using `librosa`
- 🧠 Loads a pretrained PyTorch model from `harmonianet.pth`
- 📤 Returns JSON predictions sorted by confidence
- ☁️ Deployable to Fly.io via Docker and `fly.toml`

---

## 🛠️ Environment Setup (with GPU Support)

### 1. Clone or download the backend

Make sure you have these files:

- `app.py`
- `model.py`
- `utils.py`
- `harmonianet.pth`
- `requirements.txt`
- `Dockerfile` and `fly.toml` (for deployment)

### 2. Install Python and pip (if not already installed)

To check versions:

    python --version
    pip --version

### 3. [Optional] Create a virtual environment

On macOS/Linux:

    python -m venv venv
    source venv/bin/activate

On Windows:

    python -m venv venv
    venv\Scripts\activate

### 4. Install Python dependencies

    pip install -r requirements.txt

---

## 🧪 Run Locally

    uvicorn app:app --reload

Send a test request:

    curl -X POST -F "file=@yourfile.mp3" http://localhost:8000/predict

---

## 🌍 Deployment on Fly.io

Install and log in:

    fly auth login
    fly launch

Then deploy:

    fly deploy

Once deployed, the API will be live at:

    https://<your-app-name>.fly.dev/predict

---

## 📦 Folder Structure

    harmonianet-backend/
    ├── app.py
    ├── model.py
    ├── utils.py
    ├── harmonianet.pth
    ├── Dockerfile
    ├── fly.toml
    └── requirements.txt

---

## 📬 API: `/predict`

**Method:** POST  
**Payload:** Audio file (`multipart/form-data`)  
**Returns:** JSON of genre: probability

Example:

    {
      "Jazz": 0.42,
      "Pop": 0.33,
      "Blues": 0.25
    }

---

## ⚠️ Notes

- `harmonianet.pth` must match the CNN architecture in `model.py`
- Works best with `.mp3` audio format
- Model should be trained using the FMA dataset (preprocessed to mel spectrograms)
