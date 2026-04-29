import os
import torch
import runpod
import base64
from io import BytesIO
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL

pipe = None

def load_model():
    global pipe
    if pipe is None:
        # Based on your logs, the folders are directly in /runpod-volume
        base = "/runpod-volume"
        
        print(f"✅ Confirmed: Loading Z-Image Turbo components from {base}")
        
        # Set exact paths based on your verified directory structure
        model_ckpt = os.path.join(base, "diffusion_models/z_image_turbo_bf16.safetensors")
        vae_ckpt = os.path.join(base, "vae/ae.safetensors")
        lora_path = os.path.join(base, "loras/pixel_art_style_z_image_turbo.safetensors")

        # 1. Load the main SDXL-based Pipeline
        # We use from_single_file because you have a standalone .safetensors file
        print(f"Loading main model: {model_ckpt}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_ckpt,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to("cuda")

        # 2. Load the custom VAE (ae.safetensors)
        # Custom VAEs are critical for high-quality Turbo generation
        if os.path.exists(vae_ckpt):
            print(f"Loading custom VAE: {vae_ckpt}")
            pipe.vae = AutoencoderKL.from_single_file(
                vae_ckpt, 
                torch_dtype=torch.bfloat16
            ).to("cuda")

        # 3. Apply the Pixel Art LoRA
        if os.path.exists(lora_path):
            print(f"Applying LoRA weights: {lora_path}")
            pipe.load_lora_weights(lora_path)
            
        print("🚀 Success: Model, VAE, and LoRA are fully loaded and ready!")
def handler(job):
    load_model()
    job_input = job["input"]
    
    prompt = job_input.get("prompt", "A cinematic digital art piece")
    steps = job_input.get("steps", 4)
    cfg = job_input.get("guidance_scale", 1.5)
    
    with torch.inference_mode():
        image = pipe(
            prompt=prompt, 
            num_inference_steps=steps, 
            guidance_scale=cfg
        ).images[0]
    
    # Convert image to Base64 for the API response
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}

runpod.serverless.start({"handler": handler})
