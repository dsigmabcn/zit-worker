import os
import torch
import runpod
import base64
from io import BytesIO
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL

# Model cache to prevent re-loading on every request
pipe = None

def load_model():
    global pipe
    if pipe is None:
        # 1. Define all possible root locations for your files
        # This covers the volume root, the workspace subfolder, and the standard Pod path
        possible_bases = [
            "/mnt/volume",
            "/mnt/volume/workspace",
            "/workspace"
        ]
        
        found_base = None
        target_file = "diffusion_models/z_image_turbo_bf16.safetensors"

        print("--- Diagnostic Path Search Starting ---")
        for base in possible_bases:
            full_path = os.path.join(base, target_file)
            exists = os.path.exists(full_path)
            print(f"Checking: {full_path} -> {'EXISTS' if exists else 'NOT FOUND'}")
            
            if exists:
                found_base = base
                break
        
        # 2. If nothing is found, list the directory contents to the log to debug
        if found_base is None:
            print("❌ CRITICAL ERROR: Could not find the model file in any expected path.")
            for base in possible_bases:
                if os.path.exists(base):
                    print(f"Contents of {base}: {os.listdir(base)}")
                else:
                    print(f"Directory {base} does not even exist.")
            raise FileNotFoundError("Check RunPod logs for directory listing to fix paths.")

        print(f"✅ Success! Using base path: {found_base}")
        print("--- Diagnostic Path Search Finished ---")

        # 3. Set final paths based on the found base
        model_ckpt = os.path.join(found_base, "diffusion_models/z_image_turbo_bf16.safetensors")
        vae_ckpt = os.path.join(found_base, "vae/ae.safetensors")
        lora_path = os.path.join(found_base, "loras/pixel_art_style_z_image_turbo.safetensors")

        # 4. Load the Pipeline
        print(f"Loading SDXL Pipeline from: {model_ckpt}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_ckpt,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to("cuda")

        # 5. Load VAE
        if os.path.exists(vae_ckpt):
            print(f"Loading custom VAE from: {vae_ckpt}")
            pipe.vae = AutoencoderKL.from_single_file(
                vae_ckpt, 
                torch_dtype=torch.bfloat16
            ).to("cuda")

        # 6. Load LoRA
        if os.path.exists(lora_path):
            print(f"Applying LoRA from: {lora_path}")
            pipe.load_lora_weights(lora_path)
            
        print("🚀 Model loaded and ready for inference!")
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
