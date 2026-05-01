import os
import torchvision # Must be imported before torch/diffusers to register custom operators
import torch
import runpod
import base64
import psutil
from io import BytesIO
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, ZImageInpaintPipeline

pipe = None

def configure_hf_cache():
    ''' idea is to see if the model is saved somewhere in cache
    '''
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



#def get_diagnostics():
#    """Prints system telemetry to logs to check for bottlenecks."""
#    print("--- System Diagnostics ---")
#    print(f"CPU count: {os.cpu_count()}")
#    print(f"RAM Total: {psutil.virtual_memory().total/1e9:.2f} GB")
#    print(f"RAM Available: {psutil.virtual_memory().available/1e9:.2f} GB")
#    if torch.cuda.is_available():
#        print(f"CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
#        print(f"GPU VRAM Total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
#    else:
#        print("⚠️ CUDA NOT DETECTED")
#    print("--------------------------")

def load_model():
    global pipe
    if pipe is None:
        configure_hf_cache()
        hf_repo = "Tongyi-MAI/Z-Image-Turbo"
        network_path = "/runpod-volume/models/z-image-turbo"
        use_network = os.path.exists(network_path) and any(os.scandir(network_path))
        load_source = network_path if use_network else hf_repo

        print(f"🛰️ Loading Base Pipeline from: {load_source}")
        
        # Load the base Text-to-Image pipeline
        pipe = ZImagePipeline.from_pretrained(
            load_source,
            torch_dtype=torch.bfloat16,
            local_files_only=use_network,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        pipe.to("cuda")

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

        
def handler(job):
    # Ensure model is loaded (safety check, though we call it at bottom)
    load_model()
    
    job_input = job["input"]
    prompt = job_input.get("prompt")
    steps = job_input.get("steps", 4)
    strength = job_input.get("strength", 0.7)
    seed = job_input.get("seed")
    
    # Handle Optional Inputs
    input_image_b64 = job_input.get("image")
    mask_image_b64 = job_input.get("mask_image")

    # Set generator for reproducibility
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

    with torch.inference_mode():
        # TASK 1: Inpainting
        if input_image_b64 and mask_image_b64:
            print("🖌️ Task: Inpainting")
            inpaint_pipe = ZImageInpaintPipeline.from_pipe(pipe)
            image = inpaint_pipe(
                prompt=prompt,
                image=decode_base64_to_image(input_image_b64),
                mask_image=decode_base64_to_image(mask_image_b64),
                num_inference_steps=steps,
                generator=generator
            ).images[0]

        # TASK 2: Image-to-Image
        elif input_image_b64:
            print("🖼️ Task: Image-to-Image")
            i2i_pipe = ZImageImg2ImgPipeline.from_pipe(pipe)
            image = i2i_pipe(
                prompt=prompt,
                image=decode_base64_to_image(input_image_b64),
                strength=strength,
                num_inference_steps=steps,
                generator=generator
            ).images[0]

        # TASK 3: Text-to-Image (Default)
        else:
            print("📝 Task: Text-to-Image")
            image = pipe(
                prompt=prompt,
                height=job_input.get("height", 1024),
                width=job_input.get("width", 1024),
                num_inference_steps=steps,
                guidance_scale=job_input.get("guidance", 0.0),
                generator=generator
            ).images[0]

    # Encode Result
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return {"image": base64.b64encode(buffered.getvalue()).decode("utf-8")}

# PRE-LOAD before the listener starts
load_model()
runpod.serverless.start({"handler": handler})