import os
import re
import io
import base64
import hashlib
import requests
import concurrent.futures
from typing import Dict, List
from functools import lru_cache
from fastapi import FastAPI, Request, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

API_KEY = "7777777"
MODEL_DIR = "./trocr-large-printed"
MODEL_NAME = "microsoft/trocr-large-printed"
FINGERPRINT_SERVER_URL = "https://blsautomatedbysa7tout.onrender.com/api/check_fingerprint"

# Global cache for storing processed image results
result_cache = {}

def ensure_local_trocr_model(model_dir=MODEL_DIR, model_name=MODEL_NAME):
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
        processor = TrOCRProcessor.from_pretrained(model_name)
        processor.save_pretrained(model_dir)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.save_pretrained(model_dir)
        print("Model and processor downloaded and saved locally.")

def optimize_model(model):
    """Apply optimizations to the model for faster inference on RTX A2000"""
    # Enable torch optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster computation
    
    # Convert to half precision for RTX A2000 (with Tensor Cores)
    if torch.cuda.is_available():
        model = model.half()  # Use FP16 for faster inference and memory savings
        
        # Optimize CUDA cache for RTX A2000's 6GB VRAM
        torch.cuda.empty_cache()
        # Set a reasonable memory fraction to leave room for the CUDA context
        torch.cuda.set_per_process_memory_fraction(0.85)  # Reserve 15% for system operations
    
    return model

# Initialize model with optimizations for RTX A2000
print("Initializing model for RTX A2000...")
ensure_local_trocr_model(MODEL_DIR, MODEL_NAME)
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# RTX A2000 CUDA initialization
if torch.cuda.is_available():
    # Set CUDA device properties specific to RTX A2000
    torch.cuda.empty_cache()
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Optimize for Ampere architecture (RTX A2000)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.cuda, 'amp'):
        print("AMP (Automatic Mixed Precision) is available")

# Load and optimize model
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
model = optimize_model(model)
model = model.to(device)
model.eval()  # Set to evaluation mode

# Try to JIT compile for faster inference on RTX A2000
try:
    print("Optimizing model with TorchScript for RTX A2000...")
    # Create example input for tracing (adjust dimensions as needed for your model)
    # For RTX A2000, using the exact expected input size improves performance
    dummy_batch_size = 9  # Typical batch size for captcha processing
    # Use default TrOCR image size
    height, width = 384, 384  # Adjust if your model expects different dimensions
    
    # Create dummy data that matches expected model input
    dummy_pixel_values = torch.randn(dummy_batch_size, 3, height, width, device=device).half()
    
    # Define a wrapper function that matches how you'll use the model
    def generate_wrapper(pixel_values):
        """Wrapper for generate to make it JIT compatible"""
        return model.generate(
            pixel_values,
            max_new_tokens=3,
            do_sample=False,
            num_beams=1
        )
    
    # First, try to trace the generate function
    try:
        traced_generate = torch.jit.trace(
            generate_wrapper, 
            dummy_pixel_values,
            check_trace=False  # Skip trace checking to avoid errors
        )
        model._traced_generate = traced_generate
        print("Successfully traced generate function")
    except Exception as e:
        print(f"Could not trace generate function: {e}")
        model._traced_generate = None
    
    # Also try to optimize the encoder part separately
    try:
        # Function to trace just the encoder
        def encoder_forward(pixel_values):
            return model.encoder(pixel_values=pixel_values)[0]
        
        traced_encoder = torch.jit.trace(
            encoder_forward,
            dummy_pixel_values,
            check_trace=False
        )
        model._traced_encoder = traced_encoder
        print("Successfully traced encoder function")
    except Exception as e:
        print(f"Could not trace encoder function: {e}")
        model._traced_encoder = None
        
    print("JIT optimization completed for RTX A2000")
except Exception as e:
    print(f"JIT optimization failed: {e}")
    model._traced_generate = None
    model._traced_encoder = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://appointment.thespainvisa.com","https://mauritania.blsspainglobal.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_hash(image_data: str) -> str:
    """Create a hash of the image data for caching"""
    return hashlib.md5(image_data.encode()).hexdigest()

def decode_base64_image(data_url: str) -> Image.Image:
    """Decode a base64 image string to a PIL Image"""
    if ',' in data_url:
        header, b64data = data_url.split(',', 1)
    else:
        b64data = data_url
    
    # Optimize: Use a more efficient decoding
    image_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Resize image if needed for faster processing
    # img = img.resize((224, 224), Image.LANCZOS)
    
    return img

def parallel_decode_images(images_dict: Dict[str, str]) -> List[tuple]:
    """Decode multiple images in parallel using threading"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all decode tasks
        future_to_idx = {
            executor.submit(
                decode_base64_image, 
                img_data
            ): (idx, image_to_hash(img_data)) 
            for idx, img_data in images_dict.items()
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, img_hash = future_to_idx[future]
            try:
                image = future.result()
                results.append((idx, image, img_hash))
            except Exception as e:
                print(f"Error decoding image {idx}: {e}")
                # Provide a placeholder to maintain order
                results.append((idx, None, img_hash))
    
    # Sort by index to maintain original order
    results.sort(key=lambda x: int(x[0]))
    return results

@lru_cache(maxsize=1000)
def extract_number_from_text(text: str) -> str:
    """Extract 3-digit number from text with caching"""
    numbers = re.findall(r'\b\d{3}\b', text)
    return numbers[0] if numbers else ""

def extract_3digit_numbers_batch(images, model, processor, device="cpu"):
    """Process a batch of images to extract 3-digit numbers with RTX A2000 optimizations"""
    # Filter out any None values from failed decodings
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        return [""] * len(images)
    
    # For RTX A2000: Implement efficient batching strategy
    # Process images in smaller batches if needed to avoid memory issues
    batch_size = 9  # Default batch size (9 images for captcha)
    results = []
    
    # Process in smaller batches if there are too many images
    for i in range(0, len(valid_images), batch_size):
        batch_images = valid_images[i:i+batch_size]
        
        # Preprocess batch of images
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):  # Use mixed precision on A2000
            batch = processor(batch_images, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Use JIT-compiled forward pass if available
                if hasattr(model, '_traced_forward') and model._traced_forward is not None:
                    logits = model._traced_forward(batch.pixel_values)
                    # Generate with optimized settings for A2000
                    generated_ids = model.generate(
                        batch.pixel_values,
                        max_new_tokens=3, 
                        do_sample=False,
                        num_beams=1
                    )
                else:
                    # Fall back to standard generation with optimized settings
                    generated_ids = model.generate(
                        batch.pixel_values,
                        max_new_tokens=3,
                        do_sample=False, 
                        num_beams=1
                    )
        
        # Decode the predictions
        predicted_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract 3-digit numbers and add to results
        batch_results = [extract_number_from_text(text) for text in predicted_texts]
        results.extend(batch_results)
        
        # Clear CUDA cache after each batch for RTX A2000
        if torch.cuda.is_available() and i + batch_size < len(valid_images):
            torch.cuda.empty_cache()
    
    # Account for any filtered-out None values
    if len(results) < len(images):
        # Create a list with empty strings for None images
        final_results = []
        valid_idx = 0
        for img in images:
            if img is None:
                final_results.append("")
            else:
                final_results.append(results[valid_idx])
                valid_idx += 1
        return final_results
    else:
        return results

@app.post("/")
async def solve_captcha(
    request: Request,
    background_tasks: BackgroundTasks,
    apiKey: str = Header(None)
):
    try:
        # Parse request data
        data = await request.json()
        images_dict = data.get("images", {})
        fingerprint = data.get("fingerprint", None)
        
        # Validate request
        if len(images_dict) != 9:
            return JSONResponse(
                {"status": "error", "error": "Exactly 9 images required"}, 
                status_code=400
            )
        
        # API key and fingerprint validation
        if apiKey == API_KEY:
            pass  # Skip fingerprint check with API key
        elif not apiKey:
            # Require fingerprint and check it
            if not fingerprint:
                return JSONResponse(
                    {"status": "error", "error": "Missing fingerprint"}, 
                    status_code=401
                )
            
            # Call external fingerprint server
            try:
                resp = requests.post(
                    FINGERPRINT_SERVER_URL,
                    json={"fingerprint": fingerprint},
                    timeout=5  # Reduced timeout for faster response
                )
                if resp.status_code != 200 or not resp.json().get("authorized"):
                    return JSONResponse(
                        {"status": "error", "error": "Fingerprint not authorized"}, 
                        status_code=403
                    )
            except Exception as e:
                return JSONResponse(
                    {"status": "error", "error": f"Fingerprint check failed: {str(e)}"}, 
                    status_code=500
                )
        else:
            return JSONResponse(
                {"status": "error", "error": "Invalid API key"}, 
                status_code=401
            )
        
        # Check if we can use cached results
        cached_results = {}
        need_processing = {}
        
        # First pass: Check cache for each image
        for idx, img_data in images_dict.items():
            img_hash = image_to_hash(img_data)
            if img_hash in result_cache:
                cached_results[idx] = result_cache[img_hash]
            else:
                need_processing[idx] = img_data
        
        # If all results are cached, return immediately
        if len(cached_results) == 9:
            solution = {str(i): cached_results.get(str(i), "") for i in range(9)}
            return {
                "status": "solved",
                "solution": solution
            }
        
        # Process images that need processing
        if need_processing:
            # Decode images in parallel
            decoded_images = parallel_decode_images(need_processing)
            
            # Extract images and their hashes
            images = [item[1] for item in decoded_images]
            indices = [item[0] for item in decoded_images]
            img_hashes = [item[2] for item in decoded_images]
            
            # Process the batch
            results = extract_3digit_numbers_batch(images, model, processor, device=device)
            
            # Update cache with new results
            for i, result in enumerate(results):
                idx = indices[i]
                img_hash = img_hashes[i]
                result_cache[img_hash] = result
                cached_results[idx] = result
        
        # Combine all results
        solution = {str(i): cached_results.get(str(i), "") for i in range(9)}
        
        # Clean up old cache entries in the background
        background_tasks.add_task(clean_cache)
        
        return {
            "status": "solved",
            "solution": solution
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"status": "error", "error": f"Request processing error: {str(e)}"}, 
            status_code=500
        )

def clean_cache():
    """Clean up old cache entries if cache gets too large"""
    global result_cache
    if len(result_cache) > 10000:  # Adjust threshold as needed
        # Keep only the most recent entries
        keys = list(result_cache.keys())
        for key in keys[:-5000]:  # Keep the 5000 most recent entries
            result_cache.pop(key, None)

if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    
    # For RTX A2000: Optimal server configuration
    if torch.cuda.is_available():
        worker_count = 1  # Single worker is better for GPU workloads
        # Set environment variables for CUDA optimization
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only the first GPU
        os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
        os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads
        print("RTX A2000 optimized configuration activated")
    else:
        # For CPU, use multiple workers but don't exceed CPU count
        worker_count = min(multiprocessing.cpu_count(), 4)
    
    print(f"Starting server with {worker_count} workers")
    
    uvicorn.run(
        "captcha_server:app", 
        host="0.0.0.0", 
        port=7777,
        workers=worker_count if worker_count > 1 else None,  # Don't use workers param if 1
        reload=False,
        # Optimized server settings
        loop="uvloop",
        http="httptools",
        log_level="warning",  # Reduce logging overhead
        limit_concurrency=10  # Limit concurrent connections to prevent GPU overload
    )