Runpod serverless worker for image and video generation.
Run ComfyUI (localy), but generate the images (sample them) with a GPU from Runpod

---
Based and modified from the runpod template
[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-template)](https://www.runpod.io/console/hub/runpod-workers/worker-template)

---

## Why this worker/template

1. You want to run ComfyUI locally, but generate the image/video (inference) in a more powerful or faster GPU (better that your computer)
2. Run workflows that contain different models and may overload your GPU (e.g. run image model, then Image to Video)
3. Pay only for 'generation time', not pod time

## Limitations
1. In reality, you do not only pay for generation time: machine needs to start up (load the container), download models, run, etc. There are some 'tricks' (e.g. cached models) that minimize the cold start, but to be charged 'only inference time' is not possible. 
2. Diffusers are used for inference, which eventually provide different results than using the KSampler of Comfyui
3. For the moment, limited models are suitable

## Suitable
- Z-image Turbo (more or less tested)
- Wan 2.1/2.2 (development)


## How to use it

[TO BE DONE] 



## Deploying to RunPod

There are two main ways to deploy your worker:

1.  **GitHub Integration (Recommended):**

    - Connect your GitHub repository to RunPod Serverless. RunPod will automatically build and deploy your worker whenever you push changes to your specified branch.
    - For detailed instructions on setting up the GitHub integration, authorizing RunPod, and configuring your deployment, please refer to the [RunPod Deploy with GitHub Guide](https://docs.runpod.io/serverless/github-integration).

2.  **Manual Docker Build & Push:**
    - For detailed instructions on building the Docker image locally and pushing it to a container registry, please see the [RunPod Serverless Get Started Guide](https://docs.runpod.io/serverless/get-started#step-6-build-and-push-your-docker-image).
    - Once pushed, create a new Template or Endpoint in the RunPod Serverless UI and point it to the image in your container registry.

## Further Information

- [RunPod Serverless Documentation](https://docs.runpod.io/serverless/overview)
- [Python SDK](https://github.com/runpod/runpod-python)
- [Base Docker Images](https://github.com/runpod/containers/tree/main/official-templates/base)
- [Community Discord](https://discord.gg/cUpRmau42Vd)
