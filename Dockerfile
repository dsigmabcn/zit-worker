FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404


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
