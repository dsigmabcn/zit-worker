import os
import runpod
from utils import configure_hf_cache
import sys
import argparse

# 1. Global Configuration
configure_hf_cache()
MODEL_TYPE = os.environ.get("MODEL_NAME", "tongyi-mai/z-image-turbo").lower()

# 2. Pre-initialize the engine at the module level
# This happens when the container starts, before the handler is ever called.
def initialize_engine():
    if MODEL_TYPE == "tongyi-mai/z-image-turbo":
        from zit import ZImageEngine
        engine_instance = ZImageEngine()
    elif MODEL_TYPE == "wan-ai/wan2.2-i2v-a14b-diffusers":
        from wan22_I2V import WanVideoEngine
        engine_instance = WanVideoEngine()
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    
    # Perform the heavy VRAM loading here
    engine_instance.load()
    return engine_instance

# 3. Handler
def handler(job):
    """The standard RunPod handler format."""
    try:
        job_input = job.get("input", {})
        return active_engine.execute(job_input)
    except Exception as e:
        print(f"❌ Execution Error: {str(e)}")
        return {"error": str(e), "status": "failed"}

# 4. Pod Mode
def run_pod_mode(port: int):
    """Starts a FastAPI server to mimic the RunPod serverless API."""
    print(f"🌐 Starting in POD MODE on port {port}")
    from fastapi import FastAPI, Request
    import uvicorn

    app = FastAPI(title="Engine Pod API")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_TYPE}

    @app.post("/runsync")
    async def run_sync(request: Request):
        data = await request.json()
        job_format = data if "input" in data else {"input": data}
        return handler(job_format)

    uvicorn.run(app, host="0.0.0.0", port=port)

# 5. Serverless Mode
def run_serverless_mode():
    """Starts the RunPod serverless worker."""
    print("☁️ Starting in SERVERLESS MODE")
    runpod.serverless.start({"handler": handler})

# 6. Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run engine in Serverless or Pod mode.")
    parser.add_argument("--pod_mode", action="store_true", help="Run as a persistent FastAPI web server")
    parser.add_argument("--port", type=int, default=8000, help="Port for pod_mode")
    args = parser.parse_args()

    print(f"🚀 Initializing model: {MODEL_TYPE}...")
    global active_engine
    active_engine = initialize_engine()

    if args.pod_mode:
        run_pod_mode(args.port)
    else:
        run_serverless_mode()