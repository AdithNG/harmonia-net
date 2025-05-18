# 🎵 HarmoniaNet

[🌐 Try it out here](https://harmonia-net.vercel.app/)

_A Neural Network-Based Music Genre Classifier_ 
---


## 🚀 Overview

HarmoniaNet is a full-stack web application that predicts the genre of uploaded audio tracks using a convolutional neural network trained on raw audio features. Users can contribute feedback to improve the model by verifying predictions via a Google Form.

---

## 🌟 Features

- 🎧 Upload music tracks and receive genre predictions
- ⚡ Real-time processing using a PyTorch CNN model
- 🧠 Trained on raw audio spectrograms (FMA dataset)
- 🌍 Backend hosted on **Fly.io**, frontend on **Vercel**
- 📋 Integrated Google Form for user feedback
- 🔄 Live feedback loop to improve model accuracy
- 🔐 CORS configured and memory-optimized for deployment
- 🕓 Uptime maintained using Uptime Robot

---

## 🛠 Tech Stack

| Layer      | Tech Used                        |
| ---------- | -------------------------------- |
| Frontend   | React, Framer Motion, Vercel     |
| Backend    | FastAPI, Python, PyTorch, Docker |
| ML Model   | CNN (Mel spectrogram-based)      |
| Hosting    | Fly.io (1GB VM)                  |
| Feedback   | Google Forms + Sheets            |
| Monitoring | Uptime Robot                     |

---

## 🔁 Workflow

1. User uploads an audio file (MP3/WAV)
2. The backend extracts Mel spectrograms
3. A CNN model classifies the audio into one of 16 genres
4. The frontend shows:
   - Top prediction with confidence
   - Probability breakdown
   - A **Google Form link** with the prediction pre-filled
5. User selects the correct genre and submits the form

---

## 📝 Google Form Integration

- **Question 1**: Auto-collected email
- **Question 2**: Pre-filled predicted genre (multiple choice)
- **Question 3**: User-selected correct genre (same 16-option list)
- **Responses**: Synced live to Google Sheets

---

## 📊 Data Collection and Evaluation

Once responses are collected:

- Export as CSV from Google Sheets
- Analyze accuracy with Python or Excel
- Compare predicted vs actual to generate confusion matrices
- Use data to retrain and fine-tune the model

---

## 📬 Contact

Questions, suggestions, or feedback? Submit them through the feedback form or reach out directly.

---
