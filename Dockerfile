FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN python -m pip install --upgrade pip
RUN pip install uv

# Install dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt --no-cache-dir --system

# Add files
COPY *.py /

# Run the handler
#CMD python -u /handler.py
CMD ["python", "-u", "/handler.py"]
