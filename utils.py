import os
import glob
from huggingface_hub import hf_hub_download
from io import BytesIO
from PIL import Image
import base64
import tempfile


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

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

def decode_base64_to_video_path(base64_string):
    """Saves base64 video data to a temporary file and returns the path."""
    video_data = base64.b64decode(base64_string)
    # We use delete=False so the pipeline can access the path; 
    # remember to cleanup in the handler's 'finally' block.
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video.write(video_data)
    temp_video.close()
    return temp_video.name