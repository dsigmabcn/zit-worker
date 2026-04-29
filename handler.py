import os
import torch
import runpod
import base64
from io import BytesIO
# Import the specific pipeline class for this model
from diffusers import ZImagePipeline 

pipe = None

def load_model():
    global pipe
    if pipe is None:
        # Path to the folder where you ran snapshot_download
        model_path = "/runpod-volume/models/z-image-turbo"
        
        print(f"✅ Loading Z-Image Turbo using ZImagePipeline from: {model_path}")

        # Exact implementation from official docs
        pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=False, # Set to False as per Zit docs for better stability
        ).to("cuda")

        # Optional: Apply Flash Attention for speed if using 3090/4090/A-series
        try:
            pipe.transformer.set_attention_backend("flash")
            print("⚡ Flash Attention enabled")
        except Exception as e:
            print(f"Flash Attention not available: {e}")

        # Optional: Load your Pixel Art LoRA
        lora_path = "/runpod-volume/loras/pixel_art_style_z_image_turbo.safetensors"
        if os.path.exists(lora_path):
            print(f"Applying LoRA: {lora_path}")
            pipe.load_lora_weights(lora_path)
            
        print("🚀 Success: Z-Image Pipeline is fully loaded!")

def handler(job):
    load_model()
    job_input = job["input"]
    
    prompt = job_input.get("prompt", "A cinematic digital art piece")
    # Docs recommend 9 steps for 8 forwards
    steps = job_input.get("steps", 9) 
    # Docs explicitly say guidance_scale should be 0.0 for Turbo
    cfg = job_input.get("guidance_scale", 0.0)
    
    # Generate image using the exact parameters from the Z-Image docs
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=torch.Generator("cuda").manual_seed(job_input.get("seed", 42)),
        ).images[0]
    
    # Standard Base64 return for RunPod
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}

runpod.serverless.start({"handler": handler})
