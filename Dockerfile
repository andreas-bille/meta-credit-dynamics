FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

ARG TARGET=cpu

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements*.txt ./

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install base or GPU requirements depending on TARGET
RUN if [ "$TARGET" = "gpu" ]; then \
        pip install --no-cache-dir -r requirements-gpu.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

CMD ["bash"]

EXPOSE 8888
