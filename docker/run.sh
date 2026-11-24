#!/usr/bin/env bash
# Run the enrichment crawler inside the container.
#   docker/run.sh build
#   docker/run.sh gpu-check
#   docker/run.sh test
#   docker/run.sh crawl <enrich_patent_figures.py args...>
# Host exposure: repo mounted READ-ONLY; only data/figure_enrichment is writable;
# model weights go to the named volume `patent-hf-cache` (not ~/.cache).
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=patent-figure-enrich
mkdir -p data/figure_enrichment
RUN=(docker run --rm --gpus all -v "$PWD":/work:ro
     -v "$PWD/data/figure_enrichment":/work/data/figure_enrichment
     -v /etc/ssl/certs/ca-certificates.crt:/etc/host-ca.crt:ro -v patent-hf-cache:/hf-cache -e SSL_CERT_FILE=/etc/host-ca.crt -e REQUESTS_CA_BUNDLE=/etc/host-ca.crt -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_DISABLE_XET=1 -e HF_HUB_DISABLE_PROGRESS_BARS=1
     -e PATENT_VLM_LOCAL_MODEL -e PATENT_VLM_LOCAL_REVISION)
case "${1:-}" in
  build) docker build -t "$IMG" -f docker/Dockerfile docker ;;
  gpu-check) "${RUN[@]}" --entrypoint python "$IMG" -c \
    "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))" ;;
  test) "${RUN[@]}" --entrypoint python "$IMG" -m pytest tests -q -p no:cacheprovider ;;
  crawl) shift; "${RUN[@]}" "$IMG" "$@" ;;
  *) echo "usage: $0 build|gpu-check|test|crawl ARGS"; exit 1 ;;
esac
