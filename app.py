import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from config import DEVICE, MODEL_SAVE_PATH
from data_preprocessing import test_transform
from model import BasicCNN

# Candidate locations for the trained model weights
WEIGHT_CANDIDATES = [
    MODEL_SAVE_PATH,
    "cat_dog_model.pth",
]

model: BasicCNN | None = None
loaded_weights_path: str | None = None


def find_and_load_model() -> tuple[BasicCNN, str | None]:
    """Loads BasicCNN weights if found on disk; otherwise returns an uninitialized model."""
    net = BasicCNN().to(DEVICE)
    for path in WEIGHT_CANDIDATES:
        if os.path.isfile(path):
            net.load_state_dict(torch.load(path, map_location=DEVICE))
            net.eval()
            print(f"Loaded trained weights from '{path}' on {DEVICE}")
            return net, path

    net.eval()
    print(
        f"Warning: No trained weights found at {WEIGHT_CANDIDATES}. "
        "Running with untrained weights until 'cat_dog_model.pth' is provided."
    )
    return net, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, loaded_weights_path
    model, loaded_weights_path = find_and_load_model()
    yield


app = FastAPI(
    title="Cat & Dog Detection API",
    description="Upload an image to classify whether it is a Cat or a Dog using PyTorch BasicCNN.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_home():
    """Serves the main frontend web page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


@app.get("/health")
async def health_check():
    """Returns server status and whether trained model weights are loaded."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "weights_loaded": loaded_weights_path is not None,
        "weights_path": loaded_weights_path,
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Accepts an uploaded image file and returns the Cat/Dog prediction and confidence scores."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file (JPG, PNG, WEBP).")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image file: {exc}") from exc

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized yet.")

    # Preprocess and run inference
    image_tensor = test_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    classes = ["Cat", "Dog"]
    predicted_class = classes[predicted_idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)
    cat_prob = round(probabilities[0].item() * 100, 2)
    dog_prob = round(probabilities[1].item() * 100, 2)

    return JSONResponse({
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": confidence_pct,
        "probabilities": {
            "Cat": cat_prob,
            "Dog": dog_prob,
        },
        "weights_loaded": loaded_weights_path is not None,
        "weights_path": loaded_weights_path,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

