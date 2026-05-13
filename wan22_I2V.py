import torch
import base64
import tempfile
import os
from io import BytesIO
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from utils import resolve_snapshot_path, decode_base64_to_image
from base_engine import BaseEngine
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image

class WanVideoEngine(BaseEngine):
    def __init__(self):
        super().__init__()
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return

        hf_repo = os.environ.get("MODEL_NAME", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        snapshot_path = resolve_snapshot_path(hf_repo)
              
        if snapshot_path:
            print(f"🛰️ Cached snapshot found at: {snapshot_path}")
            load_source = snapshot_path
            is_offline = True
            self._patch_missing_configs(hf_repo, snapshot_path) #running this patch because it looks cached model Runpod does not have some files
        else:
            print("ℹ️ Cache miss. Using Repo ID.")
            load_source = hf_repo
            is_offline = False


        # Load the Wan Pipeline
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            load_source,
            torch_dtype=torch.bfloat16,
            local_files_only=is_offline,
            use_safetensors=True
        )
        
        self.pipe.to("cuda")
        # Memory optimizations for large 14B models
        self.pipe.enable_attention_slicing()
        #self.pipe.enable_vae_slicing()

        print(f"🚀 Wan I2V Video Engine ({hf_repo}) successfully loaded.")

    def execute(self, job_input):
        pipeline_args = job_input.get("pipeline_args", {})

        if "seed" in pipeline_args:
            seed = pipeline_args.pop("seed")
            pipeline_args["generator"] = torch.Generator("cuda").manual_seed(seed)

        # Pop LoRA args to prevent passing them to the pipeline __call__
        pipeline_args.pop("lora_path", None)
        pipeline_args.pop("lora_strength", 1.0)

        # 2. Image Decoding (Wan 2.2 I2V requires 'image')
        if "image" in pipeline_args:
            pipeline_args["image"] = decode_base64_to_image(pipeline_args["image"])
        else:
            raise ValueError("An input 'image' is required for Wan I2V.")
        
        pipeline_args.setdefault("prompt", "")
        pipeline_args.setdefault("negative_prompt", "low quality, blurry, distorted, low resolution, noisy")

        # 3. Run Inference
        with torch.inference_mode():
            print("running inference")
            output = self.pipe(**pipeline_args)
            print("inference complete")
            video_frames = output.frames[0] 
        
        frames_b64 = []
        for i, frame in enumerate(video_frames):
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray((frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8))
            buf = BytesIO()
            frame.save(buf, format="PNG")
            frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            yield {"frame": frame_b64, "index": i, "total": len(video_frames)}

        # 4. Return the frames
        return {"frames": frames_b64}


    def _patch_missing_configs(self, hf_repo, snapshot_path):
        """to fix stupid bug when downloading the repo from hugginface"""
        if not snapshot_path:
            return

        # List of critical files needed for the MoE architecture to initialize
        missing_files = [
            "transformer_2/config.json",
            "transformer_2/diffusion_pytorch_model.safetensors.index.json"
        ]
        
        for file_path in missing_files:
            local_file = os.path.join(snapshot_path, file_path)
            if not os.path.exists(local_file):
                print(f"🛠️ Patching missing file: {file_path}")
                try:
                    subfolder, filename = os.path.split(file_path)
                    hf_hub_download(
                        repo_id=hf_repo,
                        filename=filename,
                        subfolder=subfolder,
                        local_dir=snapshot_path,
                        local_dir_use_symlinks=False,
                        cache_dir="/tmp",
                    )
                    # Verify it actually landed
                    if os.path.exists(local_file):
                        print(f"✅ Successfully patched: {file_path}")
                    else:
                        print(f"❌ Patch failed silently: {file_path}")
                except Exception as e:
                    print(f"⚠️ Failed to patch {file_path}: {e}")