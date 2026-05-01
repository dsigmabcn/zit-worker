import os
import torchvision # Must be imported before torch/diffusers to register custom operators
import torch
import runpod
import base64
import psutil
from io import BytesIO
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, ZImageInpaintPipeline
from PIL import Image

pipe = None

def configure_hf_cache():
    # RunPod Serverless Model Caching standard path
    cache_root = "/runpod-volume/huggingface-cache"
    
    # Fallback for local testing
    if not os.path.exists("/runpod-volume"):
        cache_root = "/tmp/.cache/huggingface"

    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
    
    # IMPORTANT: Force offline mode if you want to ensure NO downloads happen
    # os.environ["HF_HUB_OFFLINE"] = "1"
    print("HF_HOME =", os.environ["HF_HOME"])
    print("HF_HUB_CACHE =", os.environ["HF_HUB_CACHE"])
    print("Has /runpod-volume:", os.path.isdir("/runpod-volume"))


def load_model():
    global pipe
    if pipe is None:
        configure_hf_cache()
        
        hf_repo = "Tongyi-MAI/Z-Image-Turbo"
        
        # Check if RunPod has symlinked the cached model into the volume
        # The docs state cached models follow the HF structure here:
        cache_path = os.path.join(os.environ["HF_HUB_CACHE"], "models--Tongyi-MAI--Z-Image-Turbo")
        
        print(f"🛰️ Checking for cached model at: {cache_path}")
        
        # If the directory exists, we use local_files_only to speed up the check
        is_cached = os.path.exists(cache_path)

        pipe = ZImagePipeline.from_pretrained(
            hf_repo,
            torch_dtype=torch.bfloat16,
            local_files_only=is_cached, # Use the cache if it's there!
            use_safetensors=True
        )
        pipe.to("cuda")

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

        
def handler(job):
    # Ensure model is loaded (safety check, though we call it at bottom)
    load_model()
    
    job_input = job["input"]
    prompt = job_input.get("prompt")
    steps = job_input.get("steps", 4)
    strength = job_input.get("strength", 0.7)
    seed = job_input.get("seed")
    
    # Handle Optional Inputs
    input_image_b64 = job_input.get("image")
    mask_image_b64 = job_input.get("mask_image")

    # Set generator for reproducibility
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

    with torch.inference_mode():
        # TASK 1: Inpainting
        if input_image_b64 and mask_image_b64:
            print("🖌️ Task: Inpainting")
            inpaint_pipe = ZImageInpaintPipeline.from_pipe(pipe)
            image = inpaint_pipe(
                prompt=prompt,
                image=decode_base64_to_image(input_image_b64),
                mask_image=decode_base64_to_image(mask_image_b64),
                num_inference_steps=steps,
                generator=generator
            ).images[0]

        # TASK 2: Image-to-Image
        elif input_image_b64:
            print("🖼️ Task: Image-to-Image")
            i2i_pipe = ZImageImg2ImgPipeline.from_pipe(pipe)
            image = i2i_pipe(
                prompt=prompt,
                image=decode_base64_to_image(input_image_b64),
                strength=strength,
                num_inference_steps=steps,
                generator=generator
            ).images[0]

        # TASK 3: Text-to-Image (Default)
        else:
            print("📝 Task: Text-to-Image")
            image = pipe(
                prompt=prompt,
                height=job_input.get("height", 1024),
                width=job_input.get("width", 1024),
                num_inference_steps=steps,
                guidance_scale=job_input.get("guidance", 0.0),
                generator=generator
            ).images[0]

    # Encode Result
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return {"image": base64.b64encode(buffered.getvalue()).decode("utf-8")}

# PRE-LOAD before the listener starts
load_model()
runpod.serverless.start({"handler": handler})