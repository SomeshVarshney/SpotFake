# Screen vs Real Image Detection

A deep learning project that classifies whether an input image is:

- **Real Image**: a photograph of a real scene
- **Screen Image**: a photograph of a digital screen (monitor, phone, TV)

The model uses **transfer learning with EfficientNet-B0** and a custom-built dataset, giving strong accuracy with fast inference. It is served through a **FastAPI** backend and a lightweight web frontend.

---

## Live Demo

| Part     | Link                                  |
| -------- | ------------------------------------- |
| Frontend | https://spotfake-frontend.vercel.app           |
| API      | https://spot-fake-api.onrender.com         |
| API docs | https://spot-fake-api.onrender.com/docs    |

> The API is hosted on a free tier, so the first request after a period of inactivity can take 30-60 seconds while the server wakes up.

---

## Features

- EfficientNet-B0 transfer learning
- Two-stage fine-tuning
- Prediction with confidence score
- ROC-AUC evaluation and threshold optimization
- Benchmarking script
- GPU / CPU support
- REST API (FastAPI) with CORS enabled
- Simple web frontend with image preview and re-upload

---

## Model

| Component | Details                          |
| --------- | -------------------------------- |
| Backbone  | EfficientNet-B0 (ImageNet pretrained) |
| Classifier| Dropout (0.35) + Fully Connected layer |
| Output    | Softmax over 2 classes (0 = real, 1 = screen) |
| Loss      | CrossEntropy                     |
| Optimizer | AdamW                            |
| Scheduler | Cosine Annealing LR              |
| Input     | 224 x 224 RGB                    |

### Training Strategy

1. **Stage 1:** freeze the backbone, train the classifier head.
2. **Stage 2:** unfreeze the backbone, fine-tune the whole network at a low learning rate.

---

## Dataset

Custom dataset collected under varied lighting, camera angles, distances, reflections and display devices.

| Class  | Images |
| ------ | ------ |
| Real   | 244    |
| Screen | 209    |
| Total  | 453    |

---

## Evaluation

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 89.01% |
| Precision | 97.06% |
| Recall    | 87.57% |
| F1 Score  | 88.84% |
| ROC-AUC   | 95.80% |

## Benchmark

- Average GPU inference time: **8-10 ms**
- Throughput: **100+ FPS** (hardware dependent)
- Framework: PyTorch

---

## Project Structure

```text
spot_fake/
├── api.py                     # FastAPI server (POST /predict)
├── requirements.txt           # Full environment (training + evaluation)
├── requirements-deploy.txt    # Slim CPU-only dependencies for deployment
├── frontend/
│   └── index.html             # Single-file web UI
├── checkpoints/
│   └── best_model.pth         # Trained weights
├── src/
│   ├── benchmark.py
│   ├── config.py
│   ├── dataset.py
│   ├── engine.py
│   ├── evaluate.py
│   ├── features.py            # Handcrafted feature extraction (experimental)
│   ├── model.py
│   ├── predict.py             # Command-line prediction
│   ├── train.py
│   └── utils.py
├── images/                    # Example images
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

python -m pip install -r requirements.txt
```

---

## Usage

Run all commands from the project root.

**Train**
```bash
python src/train.py
```

**Evaluate**
```bash
python src/evaluate.py
```

**Predict (command line)**
```bash
python src/predict.py path/to/image.jpg
```
Run without an argument for interactive mode.

**Benchmark**
```bash
python src/benchmark.py
```

---

## Run the Web App Locally

**1. Start the API**
```bash
python -m pip install fastapi uvicorn python-multipart
uvicorn api:app --reload
```
The API runs at `http://127.0.0.1:8000` (interactive docs at `/docs`).

**2. Open the frontend**
```bash
cd frontend
python -m http.server 5500
```
Then open `http://127.0.0.1:5500/index.html`.

In `frontend/index.html`, `API_URL` controls which backend the page talks to.

---

## API

### `GET /`
Health check.
```json
{ "status": "ok", "device": "cpu" }
```

### `POST /predict`
Send an image as `multipart/form-data` with the field name `file`.

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@images/example.jpg"
```

Response:
```json
{
  "label": "SCREEN IMAGE",
  "confidence": 0.9712,
  "real_probability": 0.0288,
  "screen_probability": 0.9712,
  "time_ms": 41.3
}
```

---

## Deployment

| Part     | Platform | Settings |
| -------- | -------- | -------- |
| Backend  | Render (Web Service) | Build: `pip install -r requirements-deploy.txt`<br>Start: `uvicorn api:app --host 0.0.0.0 --port $PORT`<br>Env var: `PYTHON_VERSION=3.11.9` |
| Frontend | Vercel   | Root directory: `frontend`, framework preset: Other, no build command |

After the backend is live, set `API_URL` in `frontend/index.html` to the Render URL.

---

## Example Predictions

<table>
  <tr>
    <td align="center"><b>Real Image</b></td>
    <td align="center"><b>Screen Image</b></td>
  </tr>
  <tr>
    <td><img src="images/real_example.jpeg" width="300"></td>
    <td><img src="images/screen_example.jpeg" width="300"></td>
  </tr>
</table>

---

## Future Improvements

- Larger and more diverse dataset
- ConvNeXt / EfficientNet-B3 backbone
- ONNX export and TensorRT acceleration
- Mobile deployment

---

## Tech Stack

Python, PyTorch, TorchVision, Albumentations, NumPy, OpenCV, Scikit-learn, Matplotlib, FastAPI, Uvicorn, HTML/CSS/JavaScript

---

## Author

**Somesh Varshney**
