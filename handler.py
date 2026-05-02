import os
import glob
import torchvision # Must be imported before torch/diffusers
import torch
import runpod
import base64
from io import BytesIO
from threading import Lock
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, ZImageInpaintPipeline
from PIL import Image

# Global variables for persistence and performance
pipe = None
i2i_pipe = None
inpaint_pipe = None

# Thread lock to prevent concurrent requests from double-loading models
_pipe_lock = Lock()

def configure_hf_cache():
    """Sets environment variables to point to RunPod's high-speed cache."""
    cache_root = "/runpod-volume/huggingface-cache"
    if not os.path.exists("/runpod-volume"):
        cache_root = "/tmp/.cache/huggingface"

    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
    os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
    
    # Official docs suggest these for strictly cached environments
    os.environ["HF_HUB_OFFLINE"] = "1" if os.path.exists("/runpod-volume/huggingface-cache/hub") else "0"
    print(f"HF_HOME: {os.environ['HF_HOME']} | Offline Mode: {os.environ['HF_HUB_OFFLINE']}")

def resolve_snapshot_path(repo_id):
    """
    Finds the actual snapshot directory in the RunPod cache.
    Hugging Face normalizes folder names to lowercase.
    """
    # Force lowercase to match the actual file system structure
    org_repo = repo_id.replace("/", "--").lower()
    base_path = f"/runpod-volume/huggingface-cache/hub/models--{org_repo}/snapshots/*"
    print(f"🔍 Looking for snapshots in: {base_path}")
    
    snapshots = glob.glob(base_path)
    
    # Sort to get the most recent snapshot if multiple exist
    if snapshots:
        snapshots.sort(key=os.path.getmtime, reverse=True)
        return snapshots[0]
    return None

def load_model():
    global pipe, i2i_pipe, inpaint_pipe
    
    if pipe is not None:
        return

    with _pipe_lock:
        if pipe is not None:
            return

        configure_hf_cache()
        hf_repo = "Tongyi-MAI/Z-Image-Turbo"
        
        # 1. Try to find the explicit snapshot path (Doc recommendation)
        snapshot_path = resolve_snapshot_path(hf_repo)
        
        if snapshot_path:
            print(f"🛰️ Cached snapshot found at: {snapshot_path}")
            load_source = snapshot_path
            is_offline = True
        else:
            print("ℹ️ Cache miss. Using Repo ID (This will trigger a download).")
            load_source = hf_repo
            is_offline = False

        try:
            pipe = ZImagePipeline.from_pretrained(
                load_source,
                torch_dtype=torch.bfloat16,
                local_files_only=is_offline,
                use_safetensors=True
            )
            print("🚀 Model successfully loaded into VRAM.")
        except Exception as e:
            # Fallback if the snapshot path logic fails or files are corrupted
            print(f"⚠️ Snapshot load failed: {e}. Falling back to standard lookup.")
            pipe = ZImagePipeline.from_pretrained(
                hf_repo,
                torch_dtype=torch.bfloat16,
                local_files_only=False,
                use_safetensors=True
            )
        
        pipe.to("cuda")

        # Performance Knobs for Blackwell / 1024x1024 stability
        pipe.enable_attention_slicing()
        #pipe.enable_vae_slicing()
        
        # Pre-initialize specialized pipelines once
        print("🛠️ Pre-initializing specialized pipelines...")
        i2i_pipe = ZImageImg2ImgPipeline.from_pipe(pipe)
        inpaint_pipe = ZImageInpaintPipeline.from_pipe(pipe)

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

def handler(job):
    load_model()
    
    job_input = job["input"]
    prompt = job_input.get("prompt")
    steps = job_input.get("steps", 4)
    strength = job_input.get("strength", 0.7)
    seed = job_input.get("seed")
    guidance_scale = job_input.get("guidance", 0.0)
    
    input_image_b64 = job_input.get("image")
    mask_image_b64 = job_input.get("mask_image")

    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

    with torch.inference_mode():
        # --- 1. Generate Latents ---
        if input_image_b64 and mask_image_b64:
            latents = inpaint_pipe(
                prompt=prompt,
                image=decode_base64_to_image(input_image_b64),
                mask_image=decode_base64_to_image(mask_image_b64),
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent"
            ).images[0] # Note: diffusers still calls the key 'images' even for latents

        elif input_image_b64:
            latents = i2i_pipe(
                prompt=prompt,
                image=decode_base64_to_image(input_image_b64),                
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent"
            ).images[0]

        else:
            latents = pipe(
                prompt=prompt,
                height=job_input.get("height", 1024),
                width=job_input.get("width", 1024),
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent"
            ).images[0]

        # --- 2. Decode for Preview Image ---
        needs_scaling = pipe.vae.config.scaling_factor
        # Scale and decode
        decoded = pipe.vae.decode(latents / needs_scaling, return_dict=False)[0]
        image_pil = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    # --- 3. Prepare Base64 Outputs ---
    # Image Encode
    img_buf = BytesIO()
    image_pil.save(img_buf, format="PNG")
    img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8") 

    # Latent Encode (Raw Tensor)
    lat_buf = BytesIO()
    torch.save(latents.cpu(), lat_buf)
    lat_b64 = base64.b64encode(lat_buf.getvalue()).decode("utf-8")
   
    return {"image": img_b64, "latent": lat_b64}

# Listener start
runpod.serverless.start({"handler": handler})