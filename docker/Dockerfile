# Isolated environment for the figure-enrichment crawler + local VLM.
# Contains no repo code or model weights: the repo is bind-mounted read-only
# and weights live in a named Docker volume (see docker/run.sh).
FROM mirror.gcr.io/library/python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/hf-cache HF_HUB_DISABLE_XET=1

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
RUN pip install pandas openpyxl requests beautifulsoup4 lxml pillow pytest \
    "transformers>=4.49" "accelerate>=1.0" "qwen-vl-utils>=0.0.8" "bitsandbytes>=0.43"

WORKDIR /work
ENTRYPOINT ["python", "scripts/enrich_patent_figures.py"]
