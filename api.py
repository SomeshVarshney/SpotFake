import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision.models import efficientnet_b0

# ------------------------------------------------------------------
# Config (mirrors src/config.py and src/predict.py)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "checkpoints" / "best_model.pth"
IMAGE_SIZE = 224
THRESHOLD = 0.5  # same default as predict.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# Model: same architecture as src/model.py, but without downloading
# ImageNet weights (they are overwritten by your checkpoint anyway)
# ------------------------------------------------------------------
class ScreenClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 2),
        )

    def forward(self, x):
        return self.model(x)


model = ScreenClassifier().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(title="Spot Fake API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    real_prob = probs[0].item()
    screen_prob = probs[1].item()
    is_screen = screen_prob >= THRESHOLD

    return {
        "label": "SCREEN IMAGE" if is_screen else "REAL IMAGE",
        "confidence": max(real_prob, screen_prob),
        "real_probability": real_prob,
        "screen_probability": screen_prob,
        "time_ms": round(elapsed_ms, 2),
    }
