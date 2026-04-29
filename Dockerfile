FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
ENV REBUILD_VERSION=1.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

RUN python -m pip install --upgrade pip
RUN pip install uv

# Install dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install --upgrade -r /requirements.txt --no-cache-dir --system

# Add files
#ADD handler.py .
COPY handler.py /handler.py

# Run the handler
#CMD python -u /handler.py
CMD ["python", "-u", "/handler.py"]
