import os
import torchvision # Must be imported before torch/diffusers to register custom operators
import torch
import runpod
import shutil
import base64
import psutil
from io import BytesIO
from diffusers import ZImagePipeline

pipe = None

def configure_hf_cache():
    if os.path.isdir("/runpod-volume"):
        cache_root = "/runpod-volume/.cache/huggingface"
    else:
        cache_root = "/tmp/.cache/huggingface"

    hub_cache = os.path.join(cache_root, "hub")
    os.makedirs(hub_cache, exist_ok=True)

    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = hub_cache
    print("HF_HOME =", os.environ["HF_HOME"])
    print("HF_HUB_CACHE =", os.environ["HF_HUB_CACHE"])
    print("Has /runpod-volume:", os.path.isdir("/runpod-volume"))



def get_diagnostics():
    """Prints system telemetry to logs to check for bottlenecks."""
    print("--- System Diagnostics ---")
    print(f"CPU count: {os.cpu_count()}")
    print(f"RAM Total: {psutil.virtual_memory().total/1e9:.2f} GB")
    print(f"RAM Available: {psutil.virtual_memory().available/1e9:.2f} GB")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM Total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    else:
        print("⚠️ CUDA NOT DETECTED")
    print("--------------------------")

def check_disk_space(required_gb=25):
    """Checks if the local container disk has enough space for copying."""
    stats = shutil.disk_usage("/")
    free_gb = stats.free / (1024**3)
    print(f"📊 Local Disk Space: {free_gb:.2f} GB free")
    return free_gb >= required_gb

def load_model():
    global pipe
    if pipe is None:
        #get_diagnostics() # Run diagnostics on startup
        
        network_path = "/runpod-volume/models/z-image-turbo"
        local_path = "/models/z-image-turbo"
        hf_repo = "Tongyi-MAI/Z-Image-Turbo"
        
        # Determine the best source
        # Check if network drive exists AND has files in it
        use_network = os.path.exists(network_path) and any(os.scandir(network_path))
        
        load_source = hf_repo # Default fallback

        if use_network:
            print("📁 Network drive detected with content.")
            # Try to move to local SSD for speed
            load_source = network_path
        else:
            print("🌐 Network drive empty or missing. Falling back to Hugging Face download.")
            load_source = hf_repo

        # Load the pipeline
        print(f"🛰️ Loading Pipeline from: {load_source}")
        try:
            configure_hf_cache()
            pipe = ZImagePipeline.from_pretrained(
                load_source,
                torch_dtype=torch.bfloat16,
                # local_files_only must be False if we might pull from HF
                #local_files_only=(load_source != hf_repo),
                local_files_only=use_network,
                low_cpu_mem_usage=False,
            )
            
            # Critical fix for the addmm error from your previous logs
            pipe.to("cuda")
            pipe.transformer.to("cuda")
            
            print("🔥 Model is live and hot!")
        except Exception as e:
            print(f"❌ CRITICAL ERROR LOADING MODEL: {e}")
            raise e

def handler(job):
    load_model() #here?
    job_input = job["input"]

    # Lightweight health check to verify GPU visibility and driver status
    if job_input.get("check_health"):
        return {
            "status": "ready",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "FAILED",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "ram_available_gb": psutil.virtual_memory().available/1e9
        }
    
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

# Pre-load the model before starting the serverless loop
#load_model()

runpod.serverless.start({"handler": handler})
