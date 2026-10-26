#!/usr/bin/env bash
# Run once online; the Docker image then ships models/ and runs fully offline.
set -euo pipefail
pip install -q "huggingface_hub[cli]"
huggingface-cli download vinai/PhoWhisper-large --local-dir models/asr
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/base-llm
