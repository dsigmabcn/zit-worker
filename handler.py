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
        # Check both the standard RunPod mount and your custom mount path
        possible_roots = ["/runpod-volume/workspace", "/mnt/volume/workspace", "/workspace"]
        base = None
        
        print("--- Diagnostic Search Starting ---")
        for root in possible_roots:
            # We look for the main model file to verify the path
            test_path = os.path.join(root, "diffusion_models/z_image_turbo_bf16.safetensors")
            if os.path.exists(test_path):
                base = root
                print(f"✅ Found models at: {base}")
                break
            else:
                print(f"❌ Not found at: {test_path}")

        if base is None:
            print("❌ ERROR: Could not find model files. Listing /runpod-volume contents:")
            if os.path.exists("/runpod-volume"):
                print(os.listdir("/runpod-volume"))
            raise FileNotFoundError("Model files not found. Check volume mount path.")

        # Define specific file paths
        model_ckpt = os.path.join(base, "diffusion_models/z_image_turbo_bf16.safetensors")
        vae_ckpt = os.path.join(base, "vae/ae.safetensors")
        lora_path = os.path.join(base, "loras/pixel_art_style_z_image_turbo.safetensors")

        print(f"Loading Z-Image Turbo (SDXL) from {model_ckpt}...")

        # 1. Load the main Pipeline (Using from_single_file for your .safetensors structure)
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_ckpt,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to("cuda")

        # 2. Load the custom VAE (ae.safetensors)
        if os.path.exists(vae_ckpt):
            print(f"Loading custom VAE from {vae_ckpt}...")
            pipe.vae = AutoencoderKL.from_single_file(
                vae_ckpt, 
                torch_dtype=torch.bfloat16
            ).to("cuda")

        # 3. Load the Pixel Art LoRA
        if os.path.exists(lora_path):
            print(f"Applying LoRA from {lora_path}...")
            pipe.load_lora_weights(lora_path)

        print("🚀 Success: Pipeline is fully loaded and ready!")

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
