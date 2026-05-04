import sys
# Prioritize local diffusers source if available (matching your test script)
sys.path.insert(0, '/home/ohiom/diffusers/src')

from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_z_image_lora_to_diffusers
from safetensors.torch import load_file
import tempfile
import os

import os
import glob
import torchvision # Must be imported before torch/diffusers
import torch
import runpod
import diffusers
import base64
from io import BytesIO
from threading import Lock
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, ZImageInpaintPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
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

def resolve_lora_path(lora_input):
    """
    Resolves a LoRA path. If it looks like an HF repo (e.g. 'user/repo/file.safetensors'),
    it downloads it to the persistent runpod volume.
    """
    if not lora_input or not isinstance(lora_input, str):
        return None

    # If it's already an absolute path that exists, return it
    if os.path.isabs(lora_input) and os.path.exists(lora_input):
        return lora_input

    # Check for 'repo_id/filename' pattern
    parts = lora_input.split("/")
    if len(parts) >= 3:
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        
        lora_cache_dir = "/runpod-volume/loras"
        os.makedirs(lora_cache_dir, exist_ok=True)
        
        local_path = os.path.join(lora_cache_dir, parts[0], parts[1], filename)
        if not os.path.exists(local_path):
            print(f"📥 Downloading LoRA {filename} from {repo_id}...")
            return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(local_path))
        return local_path
        
    return lora_input

def load_model():
    global pipe, i2i_pipe, inpaint_pipe
    
    if pipe is not None:
        return

    with _pipe_lock:
        if pipe is not None:
            return

        configure_hf_cache()
        
        # Debug: Verify which diffusers version and path we are actually using
        print(f"📦 Diffusers version: {diffusers.__version__} from {diffusers.__file__}")

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
    # NEW: We tell the handler WHICH pipeline object to use and WHICH method
    # Defaulting to 'pipe' (txt2img) if not specified
    pipe_type = job_input.get("pipe_type", "base") 
    pipeline_args = job_input.get("pipeline_args", {})

    # Map strings to the actual global objects
    target_pipe = {
        "base": pipe,
        "i2i": i2i_pipe,
        "inpaint": inpaint_pipe
    }.get(pipe_type)

    # 1. Process specific types that can't be JSON (Generators & Images)
    if "seed" in pipeline_args:
        seed = pipeline_args.pop("seed")
        pipeline_args["generator"] = torch.Generator("cuda").manual_seed(seed)

    for img_key in ["image", "mask_image"]:
        if img_key in pipeline_args:
            pipeline_args[img_key] = decode_base64_to_image(pipeline_args[img_key])

    # 2. Handle LoRA loading/unloading
    raw_lora_input = pipeline_args.pop("lora_path", None)
    resolved_lora_path = resolve_lora_path(raw_lora_input)

    if resolved_lora_path:
        print(f"Loading LoRA weights from: {resolved_lora_path}")
        try:
            # Attempt standard load first
            pipe.load_lora_weights(resolved_lora_path, adapter_name="default")
            print("✅ LoRA weights loaded successfully.")
            print(pipe.get_list_adapters())

        except Exception as e:
            print(f"⚠️ Standard LoRA load failed: {e}")
            try:
                print("🛠️ Attempting conversion with DiT modulation key handling...")
                
                # Load the state dict
                state_dict = load_file(resolved_lora_path)
                
                # Apply the proper conversion for Z Image format
                try:
                    converted_state_dict = _convert_non_diffusers_z_image_lora_to_diffusers(state_dict)
                    print("✅ State dict converted successfully")
                except Exception as convert_e:
                    print(f"⚠️ Conversion failed ({convert_e}), attempting sanitization...")
                    # Fallback: filter problematic keys
                    converted_state_dict = {
                        k: v for k, v in state_dict.items() 
                        if "adaLN_modulation" not in k and "alpha" not in k
                    }
                
                # Save converted dict to temporary file and load it
                with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
                    tmp_path = tmp.name
                
                try:
                    from safetensors.torch import save_file
                    save_file(converted_state_dict, tmp_path)
                    pipe.load_lora_weights(tmp_path, adapter_name="default")
                    print("✅ LoRA weights loaded with conversion.")
                    print(pipe.get_list_adapters())
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    
            except Exception as patch_e:
                print(f"❌ Failed to load LoRA even with conversion: {patch_e}")
                resolved_lora_path = None

    # 2. THE DYNAMIC EXECUTION
    # This replaces your entire if/elif block. It calls whatever pipe you chose
    # with whatever arguments you sent from your local script.
    with torch.inference_mode():
        output = target_pipe(**pipeline_args, output_type="latent")
        latents = output.images[0]

        # --- 2. Decode for Preview Image ---
        needs_scaling = pipe.vae.config.scaling_factor
        print(needs_scaling)
        #needs_scaling = 1.0
        # FIX: Ensure latents has a batch dimension [1, 16, H, W]
        # If latents.ndim is 3, unsqueeze it to 4.
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        # Scale and decode
        decoded = pipe.vae.decode(latents / needs_scaling, return_dict=False)[0]
        image_pil = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    if resolved_lora_path:
        print("Unloading LoRA weights.")
        pipe.unload_lora_weights()
        print("LoRA weights unloaded.")

    # --- 3. Prepare Base64 Outputs ---
    # Image Encode
    img_buf = BytesIO()
    image_pil.save(img_buf, format="PNG")
    img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8") 

    # Latent Encode (Raw Tensor)
    latents_scaled = latents / needs_scaling
    lat_buf = BytesIO()
    torch.save(latents_scaled.cpu(), lat_buf)
    lat_b64 = base64.b64encode(lat_buf.getvalue()).decode("utf-8")
   
    return {"image": img_b64, "latent": lat_b64}

# Listener start
runpod.serverless.start({"handler": handler})