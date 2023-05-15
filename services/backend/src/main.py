from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


from PIL import Image
from io import BytesIO
import numpy as np

import torch
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "model/model.pt"
transform_path = "model/transform.pt"
mapping_path = "model/mapping.json"

model = torch.jit.load(model_path)
model.eval()
model.to("cpu")
transform = torch.jit.load(transform_path)


with open(mapping_path, "rb") as f:
    label2idx = json.load(f)
LABELS = list(label2idx.keys())


def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = load_image_into_numpy_array(await file.read())
    img = torch.from_numpy(img)
    img = transform(img).unsqueeze(0)
    log_probs = model(img)[0]
    probs = torch.exp(log_probs)
    label_probs = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return label_probs


@app.get("/")
async def read_root():
    return {"Hello": "World"}
