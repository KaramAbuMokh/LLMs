#!/bin/bash
# Builds the single-file Kaggle script of the grounding run and pushes it to the PRIVATE notebook (pushing starts the run on a free T4)
# Usage: bash push_grounding_run.sh grounding_smoke   (60 steps, a few GPU minutes)   |   bash push_grounding_run.sh grounding_full   (the real run)
set -euo pipefail  # stop on the first error
RUN_MODE="${1:?give grounding_smoke or grounding_full}"  # which run
FOLDER="$(cd "$(dirname "$0")" && pwd)"  # kaggle_finetune/retrieval_grounding_run/
KAGGLE="$HOME/.venvs/kaggle/bin/kaggle"  # Kaggle CLI (token in ~/.kaggle/access_token)
KERNEL="karamabumokh/got-gpt2-retrieval-grounding-finetune"  # the private notebook
python3 "$FOLDER/build_grounding_bundle.py" "$RUN_MODE"  # writes kernel_push/ (script + private metadata)
"$KAGGLE" kernels push -p "$FOLDER/kernel_push" --accelerator NvidiaTeslaT4 \
  || echo "The CLI printed an error, but the push may still have worked: check the status below and do NOT push again if it runs."  # seen 10-24
"$KAGGLE" kernels status "$KERNEL"  # RUNNING / QUEUED = started
