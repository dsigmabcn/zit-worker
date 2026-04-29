import os
import torch
import runpod
import base64
from io import BytesIO
from diffusers import DiffusionPipeline

# Persist the pipe in the container's global memory
pipe = None

def load_model():
    global pipe
    if pipe is None:
        # This matches the mount path you'll set in the RunPod UI
        base = "/mnt/volume"
        
        print("Initializing Z-Image Turbo from Network Volume...")
        
        # We load each component from its specific subfolder
        pipe = DiffusionPipeline.from_pretrained(
            os.path.join(base, "diffusion_models"),
            vae=os.path.join(base, "vae"),
            text_encoder=os.path.join(base, "text_encoders"),
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

        # Load the Pixel Art LoRA automatically
        lora_path = os.path.join(base, "loras/pixel_art_style_z_image_turbo.safetensors")
        if os.path.exists(lora_path):
            print("Applying Pixel Art LoRA...")
            pipe.load_lora_weights(lora_path)

def handler(job):
    load_model()
    job_input = job["input"]
    
    prompt = job_input.get("prompt", "A high-quality digital masterpiece")
    steps = job_input.get("steps", 4)
    cfg = job_input.get("guidance_scale", 1.5)
    
    with torch.inference_mode():
        image = pipe(
            prompt=prompt, 
            num_inference_steps=steps, 
            guidance_scale=cfg
        ).images[0]
    
    # Convert image to Base64 so ComfyUI can read the response
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}

runpod.serverless.start({"handler": handler})
