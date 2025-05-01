import os
import warnings
import logging

# Suppress dataloader errors and warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Load metadata ---
tracks = pd.read_csv("fma_metadata/tracks.csv", index_col=0, header=[0, 1])
genre_labels = tracks['track']['genre_top'].dropna()
genre_to_idx = {genre: idx for idx, genre in enumerate(sorted(genre_labels.unique()))}
idx_to_genre = {v: k for k, v in genre_to_idx.items()}

# --- Dataset class ---
class FMADataset(Dataset):
    def __init__(self, mel_dir, track_ids, genre_map):
        self.file_paths = []
        self.labels = []
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(sorted(genre_map.unique()))}

        for tid in track_ids:
            genre = genre_map.loc[tid]
            mel_path = os.path.join(mel_dir, f"{tid}.npy")
            if not os.path.isfile(mel_path):
                continue
            self.file_paths.append(mel_path)
            self.labels.append(self.genre_to_idx[genre])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel = np.load(self.file_paths[idx])
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label

# --- Build dataset ---
usable_ids = []
for root, _, files in os.walk("fma_small"):
    for f in files:
        if f.endswith(".mp3"):
            tid = int(f[:-4])
            if tid in genre_labels.index:
                usable_ids.append(tid)

filtered_genre_labels = genre_labels.loc[usable_ids]
track_ids = filtered_genre_labels.index.tolist()
dataset = FMADataset('mel_spectrograms', track_ids, filtered_genre_labels)

# --- Split and load ---
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

# --- Model ---
class GenreCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --- Training setup ---
num_classes = len(genre_to_idx)
model = GenreCNN(num_classes).to(device)
print(next(model.parameters()).device)  # Confirm model is on GPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# --- Training function ---
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(loader):
        inputs, labels = batch
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i == 0 or i == len(loader) - 1:
            print(f"Batch {i+1}/{len(loader)} - Loss: {loss.item():.4f}")

    return running_loss / len(loader)

# --- Training loop ---
train_losses = []
epochs = 100
for epoch in range(epochs):
    avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}\n")
    scheduler.step()

# Save model
torch.save(model.state_dict(), "harmonianet.pth")

# --- Plot loss ---
plt.plot(train_losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# --- Evaluation ---
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            correct += (pred == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.2%}')

    cm = confusion_matrix(all_labels, all_preds)
    used_genres = sorted(set(dataset.labels))
    display_labels = [idx_to_genre[i] for i in used_genres]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

evaluate(model, val_loader)
