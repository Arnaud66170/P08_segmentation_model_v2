# api/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from .inference import predict_mask

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    mask = predict_mask(image)
    return JSONResponse(content={"mask": mask.tolist()})  # ou image encodée base64
