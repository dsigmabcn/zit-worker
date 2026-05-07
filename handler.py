import os
import runpod
from utils import configure_hf_cache

# 1. Global Configuration
configure_hf_cache()
MODEL_TYPE = os.environ.get("MODEL_TYPE", "z_image").lower()

# 2. Pre-initialize the engine at the module level
# This happens when the container starts, before the handler is ever called.
def initialize_engine():
    if MODEL_TYPE == "z_image":
        from zit import ZImageEngine
        engine_instance = ZImageEngine()
    #elif MODEL_TYPE == "flux":
    #    from engines.flux import FluxEngine
     #   engine_instance = FluxEngine()
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    
    # Perform the heavy VRAM loading here
    engine_instance.load()
    return engine_instance

# Global engine variable
active_engine = initialize_engine()

# 3. The Handler (Now very lightweight)
def handler(job):
    # active_engine is already loaded and ready in VRAM
    try:
        return active_engine.execute(job["input"])
    except Exception as e:
        return {"error": str(e), "status": "failed"}

# 4. Start the serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})