#!/usr/bin/env bash
# setup_envs.sh — Recreate all conda environments from configs/
#
# Usage: bash setup_envs.sh
# Run from thesis_work/ on a fresh machine (after git clone --recursive).
#
# Environments:
#   cast_env         — CAST pipeline (GPT-4o, NWM rollout, safety margins)
#   vlajepa          — VLA-JEPA training (ViT-based NCF, DaCBaF policy)
#   hfds_env         — DROID dataset generation (HuggingFace + TFDS)
#   hf_counterfactual_env — Counterfactual generation (OpenAI batch API)
#   droid_latent_env — DROID latent manifold (VJEPAv2 encoding)
#
# CAST also has a uv-managed Python env — run `cd CAST && uv sync` separately.

set -euo pipefail

CONFIGS_DIR="$(dirname "$0")/thesis/configs"

recreate_env() {
  local name="$1" yaml="$2"
  if conda env list | grep -q "^$name "; then
    echo "  [$name] already exists — skipping (remove first to recreate)"
  else
    echo "  [$name] Creating from $yaml ..."
    conda env create -n "$name" -f "$yaml"
    echo "  [$name] Done."
  fi
}

echo "=== Recreating conda environments ==="
recreate_env "cast_env"             "$CONFIGS_DIR/cast_env.yaml"
recreate_env "vlajepa"              "$CONFIGS_DIR/vlajepa_env_full.yaml"
recreate_env "hfds_env"             "$CONFIGS_DIR/dataset.yaml"
recreate_env "hf_counterfactual_env" "$CONFIGS_DIR/counterfactual.yaml"
recreate_env "droid_latent_env"     "$CONFIGS_DIR/droid_latent_env.yaml"

echo ""
echo "=== CAST uv environment ==="
echo "  Run: cd CAST && uv sync"
echo ""
echo "=== Done! Activate with: conda activate <env_name> ==="
