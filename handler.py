import os
import torch
import runpod
import shutil
import base64
from io import BytesIO
from diffusers import ZImagePipeline

pipe = None

def load_model():
    global pipe
    if pipe is None:
        # 1. Define Paths
        network_path = "/runpod-volume/models/z-image-turbo"
        local_path = "/models/z-image-turbo" # Local container disk
        
        # 2. Copy from Network to Local SSD if not already there
        if not os.path.exists(local_path):
            print(f"📦 First time init: Copying weights to local SSD for speed...")
            os.makedirs("/models", exist_ok=True)
            # shutil.copytree is the easiest way to move the whole Diffusers folder
            shutil.copytree(network_path, local_path)
            print("✅ Copy complete.")
        else:
            print("🚀 Weights already exist on local SSD.")

        # 3. Load from LOCAL path
        print(f"🛰️ Loading Z-Image Pipeline into GPU...")
        pipe = ZImagePipeline.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=False,
        ).to("cuda")

        # Optional: Load LoRA directly from network (LoRAs are small, no need to copy)
        lora_path = "/runpod-volume/loras/pixel_art_style_z_image_turbo.safetensors"
        if os.path.exists(lora_path):
            pipe.load_lora_weights(lora_path)
            
        print("🔥 Model is live and hot!")

def handler(job):
    load_model()
    job_input = job["input"]
    
    with torch.inference_mode():
        image = pipe(
            prompt=job_input.get("prompt"),
            height=1024,
            width=1024,
            num_inference_steps=job_input.get("steps", 9),
            guidance_scale=0.0,
        ).images[0]
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return {"image": base64.b64encode(buffered.getvalue()).decode("utf-8")}

runpod.serverless.start({"handler": handler})
