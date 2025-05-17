import os
import re
import io
import base64
from typing import Dict
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

API_KEY = "7777777"
MODEL_DIR = "./trocr-base-printed"

def ensure_local_trocr_model(model_dir=MODEL_DIR):
    required_files = [
        "preprocessor_config.json",
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json"
    ]
    if not os.path.isdir(model_dir) or not all(os.path.isfile(os.path.join(model_dir, f)) for f in required_files):
        print(f"Local model not found or incomplete in '{model_dir}'. Downloading from Hugging Face Hub...")
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        processor.save_pretrained(model_dir)
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        model.save_pretrained(model_dir)
        print("Model and processor downloaded and saved locally.")

ensure_local_trocr_model(MODEL_DIR)
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

app = FastAPI()

# --- CORS FIX ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set to ["https://appointment.thespainvisa.com"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------

class ImagesRequest(BaseModel):
    images: Dict[str, str]  # index: base64 image string

def decode_base64_image(data_url: str) -> Image.Image:
    if ',' in data_url:
        header, b64data = data_url.split(',', 1)
    else:
        b64data = data_url
    image_bytes = base64.b64decode(b64data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def extract_3digit_numbers_batch(images, model, processor, device="cpu"):
    batch = processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(batch.pixel_values, max_new_tokens=6)
    predicted_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    results = []
    for text in predicted_texts:
        numbers = re.findall(r'\b\d{3}\b', text)
        results.append(numbers[0] if numbers else "")  # Return only the first 3-digit number or empty string
    return results

@app.post("/")
async def solve_captcha(
    request: Request,
    apiKey: str = Header(None)
):
    if apiKey != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        data = await request.json()
        images_dict = data.get("images", {})
        if len(images_dict) != 9:
            return JSONResponse({"status": "error", "error": "Exactly 9 images required"}, status_code=400)
        images = []
        indices = [str(i) for i in range(9)]
        for idx in indices:
            if idx not in images_dict:
                return JSONResponse({"status": "error", "error": f"Missing image index {idx}"}, status_code=400)
            images.append(decode_base64_image(images_dict[idx]))
    except Exception as e:
        return JSONResponse({"status": "error", "error": f"Invalid request: {str(e)}"}, status_code=400)

    results = extract_3digit_numbers_batch(images, model, processor, device=device)
    solution = {str(idx): value for idx, value in enumerate(results)}

    return {
        "status": "solved",
        "solution": solution
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("captcha_server:app", host="0.0.0.0", port=7777, reload=False)