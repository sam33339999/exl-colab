# Swift-1.5 Qwen3.8-27B EXL3 3.5bpw. Section 4 benchmarks were measured on this model.
# One launch:     ENV_FILE=.env.swift ./start.sh --bg
# Make it default: cp .env.swift .env
# MODEL_NAME is the /v1/models id and the weights folder name.
MODEL_REPO=sam33339999/Swift-1.5-Qwen3.8-27b-Uncensored-exl3-3.5bpw
MODEL_NAME=Swift-1.5-Qwen3.8-27b-exl3-3.5bpw

# Loaded only when DRAFT=dflash2. This is the base Qwen3.8-27B DFlash2 draft (BF16).
DRAFT_REPO=z-lab/Qwen3.8-27B-DFlash2
DRAFT_NAME=Qwen3.8-27B-DFlash2
