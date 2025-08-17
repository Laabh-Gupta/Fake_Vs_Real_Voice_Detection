# In audio_app_baseline/main.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import BaselineCNN # <-- IMPORTANT: Importing BaselineCNN
import io

# --- 1. Initialize FastAPI App and CORS ---
app = FastAPI(title="Baseline CNN - Voice Anti-Spoofing API")
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Define Constants and Load The Model ---
TARGET_SAMPLE_RATE = 16000
TARGET_LEN_SECS = 4
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
CLASS_NAMES = ["fake", "real"]
device = "cpu"

print("--- Loading BaselineCNN model ---")
model = BaselineCNN() # <-- IMPORTANT: Instantiating BaselineCNN
model.load_state_dict(torch.load("baseline_cnn_finetuned.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()
print("--- Model loaded successfully ---")


# --- 3. Define the Preprocessing Function ---
# NOTE: This version does NOT have the .resize() step needed for the ViT model
def preprocess_audio(audio_bytes: bytes):
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    target_len = TARGET_SAMPLE_RATE * TARGET_LEN_SECS
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    else:
        waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
    
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    spectrogram = mel_spectrogram_transform(waveform)
    spectrogram = spectrogram.unsqueeze(0)
    return spectrogram


# --- 4. Define the API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Baseline CNN Voice Anti-Spoofing API. Go to /docs to test."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    
    try:
        spectrogram = preprocess_audio(audio_bytes)
    except Exception as e:
        return {"error": f"Failed to process audio file: {str(e)}"}

    with torch.no_grad():
        logits = model(spectrogram.to(device))
        probabilities = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(probabilities, dim=1).item()
        prediction_class = CLASS_NAMES[prediction_idx]
        confidence = probabilities[0][prediction_idx].item()

    return {
        "filename": file.filename,
        "predicted_class": prediction_class,
        "confidence": confidence
    }