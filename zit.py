import torch
import base64
from io import BytesIO
from PIL import Image
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, ZImageInpaintPipeline
from safetensors.torch import load_file
from utils import resolve_snapshot_path, resolve_lora_path, decode_base64_to_image
from base_engine import BaseEngine

class ZImageEngine(BaseEngine):
    def __init__(self):
        super().__init__()
        self.pipe = None
        self.i2i_pipe = None
        self.inpaint_pipe = None

    def load(self):
        """Initializes the model and specialized pipelines into VRAM."""
        if self.pipe is not None:
            return

        hf_repo = "Tongyi-MAI/Z-Image-Turbo"
        snapshot_path = resolve_snapshot_path(hf_repo)
        
        if snapshot_path:
            print(f"🛰️ Cached snapshot found at: {snapshot_path}")
            load_source = snapshot_path
            is_offline = True
        else:
            print("ℹ️ Cache miss. Using Repo ID.")
            load_source = hf_repo
            is_offline = False

        try:
            self.pipe = ZImagePipeline.from_pretrained(
                load_source,
                torch_dtype=torch.bfloat16,
                local_files_only=is_offline,
                use_safetensors=True
            )
        except Exception as e:
            print(f"⚠️ Snapshot load failed: {e}. Falling back to standard lookup.")
            self.pipe = ZImagePipeline.from_pretrained(
                hf_repo,
                torch_dtype=torch.bfloat16,
                local_files_only=False,
                use_safetensors=True
            )
        
        self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()

        print("🛠️ Pre-initializing specialized pipelines...")
        self.i2i_pipe = ZImageImg2ImgPipeline.from_pipe(self.pipe)
        self.inpaint_pipe = ZImageInpaintPipeline.from_pipe(self.pipe)
        print("🚀 Z-Image Engine successfully loaded.")

    def execute(self, job_input):
        """Processes the request."""
        pipe_type = job_input.get("pipe_type", "base") 
        pipeline_args = job_input.get("pipeline_args", {})

        # Map to the correct internal pipeline
        target_pipe = {
            "base": self.pipe,
            "i2i": self.i2i_pipe,
            "inpaint": self.inpaint_pipe
        }.get(pipe_type)
        
        if target_pipe is None:
            raise ValueError(f"Invalid pipe_type: {pipe_type}. Expected 'base', 'i2i', or 'inpaint'.")

        # 1. Handle Seed/Generator
        if "seed" in pipeline_args:
            seed = pipeline_args.pop("seed")
            pipeline_args["generator"] = torch.Generator("cuda").manual_seed(seed)

        # 2. Handle Base64 Images
        for img_key in ["image", "mask_image"]:
            if img_key in pipeline_args:
                pipeline_args[img_key] = decode_base64_to_image(pipeline_args[img_key])
        # 3. Handle LoRA
        raw_lora_input = pipeline_args.pop("lora_path", None)    
        resolved_lora_path = resolve_lora_path(raw_lora_input)
        lora_strength = pipeline_args.pop("lora_strength", 1.0)

        if resolved_lora_path:
            print(f"Loading LoRA: {resolved_lora_path}")
            self.pipe.load_lora_weights(resolved_lora_path, adapter_name="lora_loaded")
            self.pipe.set_adapters(["lora_loaded"], adapter_weights=[lora_strength])

        # 4. Run Inference
        with torch.inference_mode():
            output = target_pipe(**pipeline_args, output_type="latent")
            latents = output.images[0]

            # Decode logic
            scaling_factor = self.pipe.vae.config.scaling_factor
            if latents.ndim == 3:
                latents = latents.unsqueeze(0)
            
            # Apply scaling once
            latents = latents / scaling_factor
            
            decoded = self.pipe.vae.decode(latents, return_dict=False)[0]
            image_pil = self.pipe.image_processor.postprocess(decoded, output_type="pil")[0]

        # 5. Cleanup LoRA
        if resolved_lora_path:
            self.pipe.unload_lora_weights()

        # 6. Prepare Outputs
        img_buf = BytesIO()
        image_pil.save(img_buf, format="PNG")
        img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8") 

        lat_buf = BytesIO()
        torch.save(latents.cpu(), lat_buf)
        lat_b64 = base64.b64encode(lat_buf.getvalue()).decode("utf-8")
        
        return {"image": img_b64, "latent": lat_b64}