# VoxCPM2 nano-vLLM batching dub server — everything baked in (deps + model).
# Cold-start on RunPod drops from ~237s (pip + 4.7GB model download) to just image-pull + GPU load.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    NANOVLLM_MODEL_PATH=openbmb/VoxCPM2 \
    NANOVLLM_SERVERPOOL_MAX_NUM_SEQS=32

# 1) nano-vllm-voxcpm 2.0.2 (pulls runtime deps; also drags in the wrong torch 2.12+cu130)
RUN pip install "nano-vllm-voxcpm==2.0.2"

# 2) purge the CUDA-13 libs it dragged in; reinstall driver-compatible torch 2.5.1+cu124 (cu12 cuDNN)
RUN pip uninstall -y cuda-bindings cuda-pathfinder cuda-toolkit \
        nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvshmem-cu13 || true
RUN pip install --force-reinstall "torch==2.5.1" "torchaudio==2.5.1" \
        --index-url https://download.pytorch.org/whl/cu124

# 3) flash-attn (prebuilt wheel for torch2.5/cu12/cp311) + web-server + MP3 encoder deps
RUN pip install flash-attn --no-build-isolation
RUN pip install fastapi "uvicorn[standard]" python-multipart pydantic-settings lameenc

# 4) server code
RUN git clone --depth 1 https://github.com/a710128/nanovllm-voxcpm.git /opt/nanovllm-voxcpm

# 5) bake the VoxCPM2 model (~4.7GB) into the image so no runtime download
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2')"

# 6) sanity: imports + cuda-capable build (cuda runtime check happens at pod runtime w/ GPU)
RUN python3 -c "import torch, flash_attn, nanovllm_voxcpm; print('torch', torch.__version__, 'flash', flash_attn.__version__)"

WORKDIR /opt/nanovllm-voxcpm/deployment
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
