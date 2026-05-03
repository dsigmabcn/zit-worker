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
    print(f"Looking for snapshots in: {base_path}")
    
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
    pipe_type = job_input.get("pipe_type", "base")     # Defaulting to 'pipe' (txt2img) if not specified
    pipeline_args = job_input.get("pipeline_args", {})
    if "prompt_embeds" in pipeline_args and isinstance(pipeline_args["prompt_embeds"], list):
        # 1. Convert the main embeds you SENT
        p_embeds = torch.tensor(pipeline_args["prompt_embeds"]).to(device="cuda", dtype=torch.float32)
        pipeline_args["prompt_embeds"] = p_embeds
        print(f"✅ prompt_embeds converted. Shape: {p_embeds.shape}")

        # 2. THE CRITICAL FIX: Add the pooled embeds the model REQUIRES
        # Z-Image/Flux models will crash without 'pooled_prompt_embeds'
    #    if "pooled_prompt_embeds" not in pipeline_args:
    #        batch_size = p_embeds.shape[0]
    #        # Z-Image Turbo / Flux usually uses 768 for the pooled dimension
    #        pipeline_args["pooled_prompt_embeds"] = torch.zeros((batch_size, 768), device="cuda", dtype=torch.bfloat16)
    #        print("💡 Created dummy pooled_prompt_embeds to prevent pipeline crash")
        
    target_pipe = {"base": pipe,"i2i": i2i_pipe, "inpaint": inpaint_pipe}.get(pipe_type) # Maps the pipe to use

    # generator is created in the worker
    if "seed" in pipeline_args:
        seed = pipeline_args.pop("seed")
        pipeline_args["generator"] = torch.Generator("cuda").manual_seed(seed)
    
    # Image received as data b64 (in json) converted to image for pipeline
    for img_key in ["image", "mask_image"]:
        if img_key in pipeline_args:            
            img_bytes = base64.b64decode(pipeline_args[img_key])
            pipeline_args[img_key] = Image.open(BytesIO(img_bytes)).convert("RGB")

    # EXECUTION
    with torch.inference_mode():
        output = target_pipe(**pipeline_args, output_type="latent") #target pipe: pipe, i2i_pipe or inpaint_pipe
        latents = output.images[0]
        print(latents.ndim)
        if latents.ndim == 3: #add another dim in the latent to be able to run the VAE decode in comfyui
            latents = latents.unsqueeze(0)

        needs_scaling = target_pipe.vae.config.scaling_factor      #scaling for ZIT
        latents_scaled = latents / needs_scaling                   #scaling for ZIT 
        decoded = pipe.vae.decode(latents_scaled, return_dict=False)[0]
        image_pil = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    # --- 3. Prepare Base64 Outputs ---
    # Image Encode
    img_buf = BytesIO()
    image_pil.save(img_buf, format="PNG")
    img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8") #to be sent to comfyui

    # Latent Encode (Raw Tensor)    
    lat_buf = BytesIO()
    torch.save(latents_scaled.cpu(), lat_buf)
    lat_b64 = base64.b64encode(lat_buf.getvalue()).decode("utf-8") #to be sent to comfyui
   
    return {"image": img_b64, "latent": lat_b64}

# Listener start
runpod.serverless.start({"handler": handler})