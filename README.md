# voxcpm2-dub-server
Prebuilt Docker image for the VoxCPM2 nano-vLLM batching TTS dub server (Burmese recap pipeline).
Everything baked in (torch 2.5.1+cu124, flash-attn, nano-vllm-voxcpm 2.0.2, openbmb/VoxCPM2 model).
Image: `ghcr.io/tinmaungzin/voxcpm2-dub-server:latest`. Run on a CUDA-12.x GPU; serves `/generate` on :8000.
