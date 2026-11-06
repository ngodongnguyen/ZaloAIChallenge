#!/usr/bin/env bash
# Run once online; the Docker image then ships models/ and runs fully offline.
set -euo pipefail
pip install -q "huggingface_hub"
hf download vinai/PhoWhisper-large --local-dir models/asr-base
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/base-llm
