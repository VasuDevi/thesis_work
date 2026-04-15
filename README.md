# thesis_work

Unified repository for the safety-critical VLA thesis project.

## Repository Structure

```
thesis_work/
├── CAST/                     # Safety-augmented CAST pipeline
│   ├── cast/unsafe_cf/       # Unsafe CF generation, NWM rollout, safety margins
│   ├── unsafe_cf/            # User-facing scripts
│   ├── analysis/             # Output analysis, metrics, visualization
│   ├── configs/              # data_gen.yaml + uv.lock
│   └── pyproject.toml        # uv-managed deps (run: cd CAST && uv sync)
├── thesis/                   # Core thesis code
│   ├── ncbf.py               # LatentCBC Neural CBF training
│   ├── cbf_policy.py         # DaCBaF safety policy (Taylor et al. 1912.10099)
│   ├── per_frame_nwm_gpt_batch.py  # GPT-4o-mini MCQA batch scoring for NWM frames
│   ├── configs/              # Conda env yamls for all environments
│   └── ...                   # DROID dataset generation, VLA training scripts
├── nwm/                      # Navigation World Model (CDiT-XL/2) inference code
├── setup_envs.sh             # Recreate all conda environments
├── setup_github.sh           # Finish GitHub setup (needs GITHUB_TOKEN)
└── CLAUDE.md                 # Project guidance for Claude Code
```

## HuggingFace Repositories

Large files are stored on HuggingFace (not in git):

| Repo | Contents |
|------|----------|
| [Sswa12/thesis-ncbf-checkpoints](https://huggingface.co/Sswa12/thesis-ncbf-checkpoints) | `ncbf_best_v1_nwm_only_acc884.pt`, `ncbf_best.pt`, `ncbf_final.pt` |
| [Sswa12/thesis-cast-outputs](https://huggingface.co/datasets/Sswa12/thesis-cast-outputs) | `nwm_frames_by_key.pkl`, `unsafe_action_generation_outputs.pkl`, `counterfactual_unsafe_responses.jsonl`, safety pipeline results |

Download checkpoints:
```bash
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('Sswa12/thesis-ncbf-checkpoints', 'ncbf/ncbf_best_v1_nwm_only_acc884.pt')
print(path)
"
```

Download outputs:
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Sswa12/thesis-cast-outputs', repo_type='dataset', local_dir='./cast_outputs')
"
```

## Environment Setup

```bash
# Recreate all conda environments (from configs/ yaml files)
bash setup_envs.sh

# CAST uv environment
cd CAST && uv sync

# Activate for NCF / CBF training (vlajepa env)
conda activate vlajepa

# Activate for CAST pipeline (GPT-4o, NWM rollout)
conda activate cast_env
```

## Key Pipelines

### Neural CBF (LatentCBC)
```bash
conda activate vlajepa
cd thesis_work/thesis
python ncbf.py \
    --data-pkl cast_outputs/unsafe_action_generation/unsafe_action_generation_outputs.pkl \
    --nwm-pkl  cast_outputs/safety_margins/pipeline_v1/nwm_frames_by_key.pkl \
    --nwm-only --max-safe-trajs 2000 \
    --wandb --wandb-project ncbf-cast
```

### DaCBaF Safety Policy
```bash
python cbf_policy.py \
    --ncbf-ckpt thesis_results/ncbf/ncbf_best_v1_nwm_only_acc884.pt \
    --ag-pkl    cast_outputs/unsafe_action_generation/unsafe_action_generation_outputs.pkl \
    --nwm-pkl   cast_outputs/safety_margins/pipeline_v1/nwm_frames_by_key.pkl \
    --wandb

# Render nominal vs CBF-projected trajectory in NWM:
python cbf_policy.py --render \
    --ncbf-ckpt   thesis_results/ncbf/ncbf_best_v1_nwm_only_acc884.pt \
    --policy-ckpt cbf_policy_output/cbf_policy_best.pt \
    --image       /path/to/frame.jpg \
    --waypoints   /path/to/waypoints.npy \
    --nwm-repo    nwm/ \
    --nwm-ckpt    /path/to/0100000.pth.tar \
    --output-dir  ./render_output
```

### CAST Unsafe CF Pipeline
```bash
conda activate cast_env
cd CAST

python unsafe_cf/generate_unsafe_cf.py --config configs/data_gen.yaml
python unsafe_cf/generate_unsafe_actions.py --config configs/data_gen.yaml
python unsafe_cf/generate_safety_margins.py --config configs/data_gen.yaml
```

### Per-frame GPT-4o-mini NWM Scoring
```bash
conda activate cast_env
cd thesis

python per_frame_nwm_gpt_batch.py \
    --nwm-pkl /path/to/nwm_frames_by_key.pkl \
    --output-dir ./per_frame_output \
    --api-key sk-... \
    --poll
```

## GitHub Repos

| Repo | URL |
|------|-----|
| thesis | https://github.com/VasuDevi/thesis |
| CAST fork | To be added — run `bash setup_github.sh` with `GITHUB_TOKEN=ghp_...` |
| thesis_work (this) | To be added — run `bash setup_github.sh` |
