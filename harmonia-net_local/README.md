# HarmoniaNet 🎶

A PyTorch-based CNN for music genre classification using the FMA dataset and mel spectrograms.

---

## 🔧 Environment Setup (with GPU Support)

### 1. Clone or download the project

Make sure you have all the following:

- `harmonianet.py`
- `mel_spectrograms/` folder (preprocessed)
- `fma_metadata/` and `fma_small/` folders (from the official FMA dataset)
- `requirements.txt`

### 2. Install Python and pip (if not already installed)

Ensure you’re using Python 3.9+ and pip is available in your terminal.

To check:

```bash
python --version
pip --version
```

### 3. [Optional] Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
```

### 4. Install required Python packages

```bash
pip install -r requirements.txt
```

### 5. Ensure PyTorch has CUDA (GPU) support

If your GPU is an RTX 4090 (laptop or desktop), use the CUDA 12.1 build:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 6. Verify CUDA is working

In a Python shell, check:

```bash
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## 🚀 Running the Code

Once everything is set up and the mel spectrograms are already generated:

python3 harmonianet.py (or py harmonianet.py as in my case)

This will:

- Train the CNN on the FMA dataset
- Show a training loss curve
- Print validation accuracy
- Display a confusion matrix of genre predictions

---

## 🗂 Contents

- harmonianet.py – Full training pipeline
- mel_spectrograms/ – Preprocessed spectrograms
- fma_small/ – Audio files (from FMA)
- fma_metadata/ – Metadata files
- harmonianet.pth – Trained model weights (optional)
- requirements.txt – Dependencies

---

## 💡 Notes

- If you're using a laptop, ensure you're plugged into power and running on High Performance mode.
- The training pipeline automatically uses the GPU if CUDA is available.

---

## 📦 Dataset Source

FMA: Free Music Archive Dataset  
https://github.com/mdeff/fma
