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

class WanVideoEngine(BaseEngine):
    def __init__(self):
        super().__init__()
        self.pipe = None

    def load(self):
        """Initializes the Wan Image-to-Video model into VRAM."""
        if self.pipe is not None:
            return

        # Model ID for Wan 2.1 I2V (e.g., 14B-720P-Diffusers or 1.3B-480P-Diffusers)
        hf_repo = os.environ.get("MODEL_NAME", "Wan-Video/Wan2.1-I2V-14B-720P-Diffusers")
        snapshot_path = resolve_snapshot_path(hf_repo)
        
        self._patch_missing_configs(hf_repo, snapshot_path) #running the stupid patch
        
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
        # Memory optimizations for large 14B models
        self.pipe.enable_attention_slicing()
        #self.pipe.enable_vae_slicing()

        print(f"🚀 Wan I2V Video Engine ({hf_repo}) successfully loaded.")

    def execute(self, job_input):
        """Processes the Image-to-Video generation request."""
        pipeline_args = job_input.get("pipeline_args", {})

        #fps = pipeline_args.pop("fps", 16)

        if "seed" in pipeline_args:
            seed = pipeline_args.pop("seed")
            pipeline_args["generator"] = torch.Generator("cuda").manual_seed(seed)

        # Pop LoRA args to prevent passing them to the pipeline __call__
        pipeline_args.pop("lora_path", None)
        pipeline_args.pop("lora_strength", 1.0)

        # 2. Image Decoding (Wan 2.1 I2V requires 'image')
        if "image" in pipeline_args:
            pipeline_args["image"] = decode_base64_to_image(pipeline_args["image"])
        else:
            raise ValueError("An input 'image' is required for Wan I2V.")
        
        pipeline_args.setdefault("prompt", "")
        pipeline_args.setdefault("negative_prompt", "low quality, blurry, distorted, low resolution, noisy")

        # 3. Run Inference
        with torch.inference_mode():
            # Wan I2V returns frames[0] as a list of PIL images representing the video
            output = self.pipe(**pipeline_args)
            video_frames = output.frames[0] 

        '''
        # 5. Safe Video Export
        # We use delete=False to ensure the file exists for reading
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
                # Note: Wan 2.1 is optimized for 16fps
                export_to_video(video_frames, tmp_path, fps=fps)
            
            with open(tmp_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Return the first frame as a preview image
        img_buf = BytesIO()
        video_frames[0].save(img_buf, format="PNG")
        img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8") 
        
        return {"video": video_b64, "image": img_b64}'''
        frames_b64 = []
        for frame in video_frames:
            buf = BytesIO()
            # Using PNG for lossless quality
            frame.save(buf, format="PNG")
            frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

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
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    print(f"⚠️ Failed to patch {file_path}: {e}")