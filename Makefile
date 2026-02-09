IMAGE_CPU ?= dl:cpu
IMAGE_GPU ?= dl:gpu
PORT ?= 8888

PWD := $(shell pwd)
UID := $(shell id -u)
GID := $(shell id -g)

.DEFAULT_GOAL := help

.PHONY: help build-cpu build-gpu run-cpu run-gpu run-cpu-jupyter run-gpu-jupyter test-cpu test-gpu

help:
	@echo "Targets:"
	@echo "  build-cpu   - Build CPU image (tag: $(IMAGE_CPU))"
	@echo "  build-gpu   - Build GPU image (tag: $(IMAGE_GPU))"
	@echo "  run-cpu     - Run CPU container with repo mounted"
	@echo "  run-gpu     - Run GPU container with repo mounted"
	@echo "  run-cpu-jupyter - Run CPU container and launch Jupyter on port $(PORT)"
	@echo "  run-gpu-jupyter - Run GPU container and launch Jupyter on port $(PORT)"
	@echo "  test-cpu    - Run pytest inside CPU container"
	@echo "  test-gpu    - Run pytest inside GPU container"
	@echo ""
	@echo "Override tags, e.g.: make build-cpu IMAGE_CPU=dl:dev"

build-cpu: Dockerfile requirements.txt
	docker build -t $(IMAGE_CPU) --build-arg TARGET=cpu .

build-gpu: Dockerfile requirements-gpu.txt requirements.txt
	docker build -t $(IMAGE_GPU) --build-arg TARGET=gpu .

run-cpu:
	docker run -it --rm \
	  -v "$(PWD)":/workspace \
	  -w /workspace \
	  -u $(UID):$(GID) \
	  $(IMAGE_CPU) bash

run-gpu:
	docker run --gpus all -it --rm \
	  -v "$(PWD)":/workspace \
	  -w /workspace \
	  -u $(UID):$(GID) \
	  $(IMAGE_GPU) bash

run-cpu-jupyter:
	docker run -it --rm \
	  -v "$(PWD)":/workspace \
	  -w /workspace \
	  -p $(PORT):$(PORT) \
	  -u $(UID):$(GID) \
	  $(IMAGE_CPU) bash -lc "python -m ipykernel install --sys-prefix --name container --display-name 'Python (Container venv)' >/dev/null 2>&1 || true; jupyter lab --ip 0.0.0.0 --port $(PORT) --no-browser --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=True --ServerApp.allow_origin='*'"

run-gpu-jupyter:
	docker run --gpus all -it --rm \
	  -v "$(PWD)":/workspace \
	  -w /workspace \
	  -p $(PORT):$(PORT) \
	  -u $(UID):$(GID) \
	  $(IMAGE_GPU) bash -lc "python -m ipykernel install --sys-prefix --name container --display-name 'Python (Container venv)' >/dev/null 2>&1 || true; jupyter lab --ip 0.0.0.0 --port $(PORT) --no-browser --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=True --ServerApp.allow_origin='*'"

test-cpu: build-cpu
	docker run --rm \
	  -v "$(PWD)":/workspace \
	  -w /workspace \
	  -u $(UID):$(GID) \
	  $(IMAGE_CPU) bash -lc "pytest -q"

test-gpu: build-gpu
	docker run --gpus all --rm \
	  -v "$(PWD)":/workspace \
	  -w /workspace \
	  -u $(UID):$(GID) \
	  $(IMAGE_GPU) bash -lc "pytest -q"
