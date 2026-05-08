import torch
import base64
import tempfile
import os
from io import BytesIO
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from utils import resolve_snapshot_path, decode_base64_to_image
from base_engine import BaseEngine

class WanVideoEngine(BaseEngine):
    def __init__(self):
        super().__init__()
        self.pipe = None

    def load(self):
        """Initializes the Wan Image-to-Video model into VRAM."""
        if self.pipe is not None:
            return

        # Model ID for Wan 2.1 I2V (e.g., 14B-720P-Diffusers or 1.3B-480P-Diffusers)
        hf_repo = os.environ.get("WAN_MODEL_ID", "Wan-Video/Wan2.1-I2V-14B-720P-Diffusers")
        snapshot_path = resolve_snapshot_path(hf_repo)
        
        if snapshot_path:
            print(f"🛰️ Cached snapshot found at: {snapshot_path}")
            load_source = snapshot_path
            is_offline = True
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

        print(f"🚀 Wan I2V Video Engine ({hf_repo}) successfully loaded.")

    def execute(self, job_input):
        """Processes the Image-to-Video generation request."""
        pipeline_args = job_input.get("pipeline_args", {})

        # 1. Handle Seed/Generator
        if "seed" in pipeline_args:
            seed = pipeline_args.pop("seed")
            pipeline_args["generator"] = torch.Generator("cuda").manual_seed(seed)

        # 2. Handle Conditioning Image (Required for I2V)
        if "image" in pipeline_args:
            pipeline_args["image"] = decode_base64_to_image(pipeline_args["image"])
        
        # Ensure prompt is present (even if empty)
        if "prompt" not in pipeline_args:
            pipeline_args["prompt"] = ""

        # 3. Run Inference
        with torch.inference_mode():
            # Wan I2V returns frames[0] as a list of PIL images representing the video
            output = self.pipe(**pipeline_args)
            video_frames = output.frames[0] 

        # 4. Prepare Outputs
        # Export generated frames to a temporary MP4 file
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            export_to_video(video_frames, tmp.name, fps=16)
            with open(tmp.name, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Return the first frame as a preview image
        img_buf = BytesIO()
        video_frames[0].save(img_buf, format="PNG")
        img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8") 
        
        return {"video": video_b64, "image": img_b64}