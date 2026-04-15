# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Goals

1. **Get CAST pipeline working** — `CAST/` uses `uv` and GCP/Gemini for counterfactual label generation on navigation trajectories.
2. **Update CAST counterfactual generation with unsafeness-prioritizing prompts** — draw from the ideas in `thesis/counterfactual/counterfactual_generation/prompts.py` (scene analysis, violation types, kinematic opportunity detection) but adapt them to CAST's navigation domain (atomic actions: `turn left`, `turn right`, `go forward`, `stop`, `adjust left`, `adjust right`) and preserve CAST's existing input/output format. New scripts go in `CAST/`.
3. **Enable GPT-4o as an alternative backend** for CAST counterfactual generation (see below).

**Always ask before `git push`.** Otherwise, work autonomously.

---

## Repository Layout

```
thesis_work/
├── atomic_datasets/             # Downloaded CAST atomic policy datasets (TFDS TFRecord format)
│   ├── atomic_forward_dataset/1.0.0/*.tfrecord   (256 shards)
│   ├── atomic_turn_left_dataset/
│   ├── atomic_turn_right_dataset/
│   ├── atomic_stop_dataset/
│   ├── atomic_adjust_left_dataset/
│   └── atomic_adjust_right_dataset/
├── atomic_dataset.tar.gz        # Compressed archive of the above
├── CAST/                        # CAST pipeline (uv-managed, Gemini Batch API / GCP)
│   ├── cast/data/
│   │   ├── generate_cast_dataset.py   # Pipeline entry point
│   │   ├── build_atomic_dataset.py    # Atomic dataset builder
│   │   ├── prompts/                   # Prompt text files (loaded at runtime)
│   │   │   ├── cf_step_prompt.txt     # ← target for unsafeness prompt update
│   │   │   ├── filter_step_prompt.txt
│   │   │   ├── hindsight_describe_prompt.txt
│   │   │   └── hindsight_summarize_prompt.txt
│   │   ├── schemas/                   # JSON response schemas for each step
│   │   │   └── cf_step_schema.json    # Defines: unique_id + counterfactual_action_instruction[]
│   │   └── utils/
│   │       ├── common.py              # GCP/Gemini batch infra, base_actions list, make_request()
│   │       ├── counterfactual.py      # counterfactual_propose() — builds requests, runs job
│   │       ├── atomic_decomposition.py  # discretize_trajectory() — yaw/distance → atomic label
│   │       ├── hindsight_labeling.py
│   │       ├── filtering.py
│   │       └── action_generation.py
│   ├── cast/data/conversion/    # TFDS builders (cast_counterfactual, cast_filtered)
│   ├── cast/atomic_policy/      # Atomic policy training
│   ├── cast-vla/                # VLA training submodule
│   ├── configs/
│   │   └── data_gen.yaml        # Main config template (fill in paths, GCP project, model)
│   └── pyproject.toml           # uv deps; run `uv sync` from CAST/
└── thesis/                      # Separate thesis pipeline — see thesis/CLAUDE.md
    └── counterfactual/counterfactual_generation/prompts.py  # Source of unsafeness ideas
```

---

## CAST Pipeline

### Setup
```bash
cd CAST
uv sync
gcloud auth application-default login
gcloud auth application-default set-quota-project <your-project>
```

Dataset format (GNM-style): `root_dir/.../traj_name/{0.jpg, 1.jpg, ..., N.jpg, traj_data.pkl}`  
`traj_data.pkl` must contain `position` (Nx2 array) and `yaw` (N array) keys used by `discretize_trajectory()`.

### Running
```bash
cd CAST
python cast/data/generate_cast_dataset.py --hindsight-step  --config-path configs/data_gen.yaml
python cast/data/generate_cast_dataset.py --filter-step     --config-path configs/data_gen.yaml
python cast/data/generate_cast_dataset.py --cf-step         --config-path configs/data_gen.yaml
python cast/data/generate_cast_dataset.py --action-gen-step --config-path configs/data_gen.yaml
python cast/data/generate_cast_dataset.py --run-all         --config-path configs/data_gen.yaml

# Atomic policy
python cast/data/build_atomic_dataset.py --config-path configs/data_gen.yaml
./cast/atomic_policy/data/split_atomic_dataset.sh
python cast/atomic_policy/train_atomic.py --config configs/atomic_model.yaml

# TFDS conversion
cd cast/data/conversion/cast_counterfactual && tfds build [--overwrite]
cd cast/data/conversion/cast_filtered && tfds build [--overwrite]
```

### CAST Architecture

The pipeline in `generate_cast_dataset.py` calls step functions that delegate to `cast/data/utils/`:

| Step | Utility function | Key files |
|------|-----------------|-----------|
| hindsight | `hindsight_describe`, `hindsight_summarize` | `hindsight_labeling.py` |
| filter | `filtering()` | `filtering.py` |
| **counterfactual** | `counterfactual_propose()` | **`counterfactual.py`** |
| action gen | `generate_actions()` | `action_generation.py` |

**Atomic actions** (`common.py::base_actions`):  
`["turn left", "turn right", "go forward", "stop", "adjust left", "adjust right"]`  
These come from `discretize_trajectory()` in `atomic_decomposition.py`, which classifies each trajectory segment by yaw delta and distance using thresholds from `data_gen.yaml` (`min_turn_thres`, `min_forward_thres`, `min_adjust_thres`, `max_adjust_thres`).

**CF step — keyframe strategy**: CAST uses the VLM at atomic action decision points (keyframes) to solve the *posterior collapse* problem — without counterfactuals, the robot ignores language and follows visual shortcuts. At each keyframe the VLM receives the current observation image and the action the robot *actually* took, then proposes plausible alternative actions with different instructions (e.g., "robot moved toward Red Chair → counterfactual: 'Move toward the Blue Table'"). This is the core mechanism for generating negative instruction/action pairs.

**CF step data flow** (`counterfactual.py::counterfactual_propose`):
1. Loads filtered hindsight output from `{output_dir}/filter/filter/filter_responses.jsonl`
2. Calls `discretize_trajectory()` per trajectory to get atomic segments
3. For each segment (keyframe): formats prompt with `{labels}` (cumulative actions so far), `{instructions}`, `{curr_atomic_action}` (what the robot did), `{base_actions}`, `{unique_id}`; attaches the segment's first image (GCS URI)
4. Uploads batch JSONL to GCS → runs Vertex batch job → saves parsed responses to `{output_dir}/counterfactual/counterfactual/counterfactual_responses.jsonl`

**CF step schema** (`cf_step_schema.json`): Each response must contain:
- `unique_id` (string)
- `counterfactual_action_instruction` (array of objects with `counterfactual_action`, `counterfactual_instruction`, `reasoning`)

---

## GPT-4o Backend for CF Step

The existing `counterfactual_propose()` uses the Vertex AI Batch API. To add GPT-4o support:

- Add `use_openai: true` and `openai_api_key: "..."` fields to `data_gen.yaml`
- New file `CAST/cast/data/utils/counterfactual_openai.py` should mirror `counterfactual.py` but use the **OpenAI Responses API** (`client.responses.create`), not Chat Completions
- Images must be base64-encoded inline (GPT-4o does not accept GCS URIs); load locally from `traj_path` using `load_image()` from `common.py`
- Prompt interpolation, schema, and output format must stay identical to the Gemini path
- `generate_cast_dataset.py` should route to the OpenAI variant when `config.get("use_openai")` is true

**Responses API call pattern:**
```python
from openai import OpenAI
client = OpenAI(api_key=config["openai_api_key"])
response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": modified_prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64_image}"},
            ],
        }
    ],
    text={"format": {"type": "json_schema", "name": "cf_response", "schema": schema}},
)
result = json.loads(response.output_text)
```

For batched processing, collect all requests and submit via `client.responses.create` in a loop (the Responses API does not have a separate Batch endpoint like Chat Completions). Parse `response.output_text` for each. Response parsing differs from Vertex: no `candidates[0]["content"]["parts"][0]["text"]` nesting — the result is directly in `response.output_text`.

---

## Unsafe Counterfactual Alternatives (Separate Scripts)

**Do not modify any existing CAST files.** All unsafe CF work is isolated in `CAST/unsafe_cf/`:

```
CAST/unsafe_cf/                  ← user-facing scripts (run from CAST/ directory)
├── generate_unsafe_cf.py        Step 1: GPT-4o proposes unsafe CF alternatives
├── generate_unsafe_actions.py   Step 2: atomic policy generates waypoints for CF actions
└── download_gnm_data.py         Dataset downloader

CAST/cast/unsafe_cf/             ← Python package (part of the cast package)
├── __init__.py
├── counterfactual_unsafe.py     Orchestration (one GPT-4o call per chunk)
├── counterfactual_openai.py     GPT-4o Responses API backend
├── unsafe_action_generation.py  Atomic-policy waypoint generation for unsafe CFs
├── prompts/
│   └── cf_unsafe_prompt.txt     Three-phase unsafe prompt
├── schemas/
│   └── cf_unsafe_schema.json    Response schema (divergence_segment_index, failure_type)
└── conversion/
    └── cast_unsafe_cf/
        └── cast_unsafe_cf_dataset_builder.py   TFDS builder
```

**GNM data annotation — what exists vs what CAST generates:**
- Raw `traj_data.pkl` contains **only** `position` (Nx2) and `yaw` (N). No language instructions, no atomic labels.
- **Atomic labels** (turn left, go forward, etc.) are deterministically derived from position+yaw by `atomic_decomposition.py`.
- **Language instructions** are generated by the CAST hindsight step (Gemini VLM on pairwise frames).
- The `atomic_datasets/` TFRecords at `/home/vasu/thesis_work/atomic_datasets/` have `language_instruction` = the atomic label string only (e.g. "go forward"), not rich natural language.
- You cannot shortcut the pipeline using `atomic_datasets/` TFRecords — they lack images and hindsight instructions required by `generate_unsafe_cf.py`.

**Running the unsafe CF pipeline:**
```bash
cd CAST

# 1. Download GNM trajectories (first time only)
python unsafe_cf/download_gnm_data.py --dataset go_stanford2 --out-dir /path/to/gnm_data
# Set dataset_path in data_gen.yaml to the downloaded directory

# 2. Run hindsight + filter steps (prerequisite — generates language instructions)
python generate_cast_dataset.py --hindsight-step --config configs/data_gen.yaml
python generate_cast_dataset.py --filter-step    --config configs/data_gen.yaml

# 3. Run unsafe CF generation (GPT-4o picks divergence + proposes unsafe actions)
python unsafe_cf/generate_unsafe_cf.py --config configs/data_gen.yaml
# Output: {output_dir}/counterfactual_unsafe/counterfactual_unsafe/
#           counterfactual_unsafe_responses.jsonl   ← one record per trajectory chunk
#           cast_cf_unsafe_responses.pkl

# With explicit API key override:
python unsafe_cf/generate_unsafe_cf.py --config configs/data_gen.yaml --openai-api-key sk-...

# 4. Run unsafe action generation (atomic policy generates waypoints for each CF action)
python unsafe_cf/generate_unsafe_actions.py --config configs/data_gen.yaml
# Output: {output_dir}/unsafe_action_generation/unsafe_action_generation_outputs.pkl
# Requires action_model section in data_gen.yaml (same as --action-gen-step)

# 5. Build TFDS dataset
# Edit DATASET_ROOT_PATH and UNSAFE_ACTION_GEN_DIR in:
#   cast/unsafe_cf/conversion/cast_unsafe_cf/cast_unsafe_cf_dataset_builder.py
cd cast/unsafe_cf/conversion/cast_unsafe_cf && tfds build [--overwrite]
```

`unsafe_cf/generate_unsafe_cf.py` loads the same `data_gen.yaml` config. Add these keys for the unsafe pipeline:
```yaml
openai_api_key: "sk-..."        # or set OPENAI_API_KEY env var
openai_model: "gpt-4o"          # optional, default gpt-4o
context_mode: "segment_starts"  # "segment_starts" (default) or "full_images"
                                 #   segment_starts: 1 image per segment (keyframe at atomic action start)
                                 #     → 1:1 with all_labels, most token-efficient, closest to CAST approach
                                 #   full_images: all subsampled frames across chunk range
                                 #     → richer context, higher token cost
cf_unsafe_prompt: "unsafe_cf/prompts/cf_unsafe_prompt.txt"   # optional, auto-resolved if omitted
cf_unsafe_schema: "unsafe_cf/schemas/cf_unsafe_schema.json"  # optional, auto-resolved if omitted
request_delay: 1.0              # seconds between API calls
save_interval: 10               # checkpoint every N records
```

**How CAST originally handled context** (for reference):
- Original CF step: sends **one image** per request (start frame of current segment only), proposes alternatives for that segment independently — no divergence index
- Hindsight step: sends **pairwise frames** (current + previous) — not the full trajectory
- Our unsafe pipeline improves on this: sends images for the whole chunk so GPT-4o can pick the optimal divergence point across the full trajectory

**Dataset roles and language instruction fields:**

| Dataset | `language_instruction` field | Source |
|---------|------------------------------|--------|
| `atomic_datasets/` TFRecords | One of 6 atomic label strings (e.g. `"go forward"`) | `build_atomic_dataset.py` — NOT rich NL |
| `cast_filtered` TFDS | Up to 10 rich NL strings per episode (hindsight "best" + "new") | CAST hindsight + filter pipeline |
| `cast_counterfactual` TFDS | Single CF instruction string replicated across all steps | CAST per-segment CF pipeline |
| `cast_unsafe_cf` TFDS (future) | Single unsafe CF instruction (`counterfactual_instruction` field) | `unsafe_cf/generate_unsafe_cf.py` output |

- `/home/vasu/thesis_work/atomic_datasets/` — TFDS TFRecords for **atomic policy training** (`train_atomic.py`) and VLA training mix-in. Pre-built; do not re-derive.
- **Raw GNM trajectories** (GoStanford2, RECON) — input to `generate_cast_dataset.py` and `unsafe_cf/generate_unsafe_cf.py`. Download with `unsafe_cf/download_gnm_data.py`. These are NOT the atomic_datasets.
- **VLA training recipe** (thesis extension of CAST):
  ```
  cast_filtered TFDS          ← safe trajectories with rich NL instructions
  + atomic_datasets/          ← atomic policy training + VLA breadth
  + cast_counterfactual TFDS  ← safe per-segment CFs (posterior collapse fix)
  + cast_unsafe_cf TFDS       ← NEW: unsafe CFs at safety-critical moments (thesis)
  ```
  Pipeline: `generate_unsafe_cf.py` → JSONL → `generate_unsafe_actions.py` → waypoints pkl → `cast_unsafe_cf_dataset_builder.py` → TFDS.

**Design decision — why full trajectory for unsafe CF, not per-atomic:**

| | Original CAST (per-segment) | `unsafe_cf` (full trajectory) |
|---|---|---|
| Images sent | 1 frame | N keyframes (one per segment) |
| Divergence choice | None — all segments queried | GPT-4o picks the critical moment |
| CF quality | Arbitrary | Grounded in visible hazards |
| Failure type label | None | `physical` or `semantic` |
| Thesis fit | Posterior collapse fix | Safety-critical moment identification |

Use **`unsafe_cf`** for unsafe training data. For safe CF data (breadth), use the **original CAST pipeline** (`generate_cast_dataset.py --cf-step`) unchanged — it already generates good safe alternatives per segment at scale. Do not run a per-atomic version of the unsafe pipeline; it would produce low-quality unsafe examples without scene context.

**Prompt design for `cf_unsafe_prompt.txt`** — draw ideas from `thesis/counterfactual/counterfactual_generation/prompts.py` but adapt to navigation:
- **Keep CAST's template variables**: `{labels}`, `{instructions}`, `{curr_atomic_action}`, `{base_actions}`, `{unique_id}`
- **Output format**: use a new `cf_unsafe_schema.json` (same shape as `cf_step_schema.json` but with an added required field `failure_type` on each item — see schema below). `counterfactual_action` must be one value from `base_actions`; `counterfactual_instruction` ≤ 10 words.
- **Description / scene analysis phase** (before proposing counterfactuals, the prompt must instruct the model to analyze):
  1. **Collidable objects**: enumerate every object visible in the scene that the robot could physically collide with — walls, furniture, doors, people, low obstacles, protruding edges. For each, note its position relative to the robot and whether it lies along a plausible navigation path.
  2. **Instruction–action alignment**: state what `{instructions}` asked the robot to do, which actions from `{labels}` are consistent with that intent, what landmark or direction the instruction implies from the current observation, and whether `{curr_atomic_action}` aligns or conflicts. This grounding is required before choosing a failure mode.
  3. Narrow passages, drop-offs, and boundary edges visible in the image.
  4. Environment classification (corridor, office, kitchen, outdoor path, etc.) to inform which hazard to prioritize.

- **Failure modes** — the prompt must instruct the model to commit to exactly one of these two categories per counterfactual:
  - **`physical`** — the proposed action causes direct physical harm: the robot collides with a wall, obstacle, person, or boundary edge, or drives into a drop-off. The unsafe action is chosen because it steers into a collidable object identified in the scene analysis. Example: robot near a wall to the left → propose `adjust left` to drive into it.
  - **`semantic`** — the proposed action contradicts the semantic intent of the instruction relative to what is visible in the observation. The instruction implies a specific direction or landmark; the counterfactual action moves the opposite way or toward the wrong landmark. Example: instruction is "turn toward the door", door is visible to the right in the image → propose `turn left` (away from the door). The action itself may be physically safe but is semantically wrong.

- **`failure_type` output field**: each item in `counterfactual_action_instruction` must include `"failure_type": "physical" | "semantic"`. This is required in `cf_unsafe_schema.json`. Schema shape:
  ```json
  {
    "type": "object",
    "required": ["unique_id", "counterfactual_action_instruction"],
    "properties": {
      "unique_id": {"type": "string"},
      "counterfactual_action_instruction": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["counterfactual_action", "counterfactual_instruction", "reasoning", "failure_type"],
          "properties": {
            "counterfactual_action":      {"type": "string"},
            "counterfactual_instruction": {"type": "string"},
            "reasoning":                  {"type": "string"},
            "failure_type":               {"type": "string", "enum": ["physical", "semantic"]}
          }
        }
      }
    }
  }
  ```

- **Kinematic opportunity**: a run of 4+ same-direction labels = robot committed to that side; overdrive → physical collision

---

## Analysis Folder (`CAST/analysis/`)

Scripts for serializing outputs, computing metrics, and generating figures. **Do not modify any existing CAST files.**

```
CAST/analysis/
├── reconstruct_episodes.py   Bootstrap GNM trajectories from atomic TFRecords (no raw download needed)
├── serialize_outputs.py      Convert PKL action outputs → JSONL (numpy-safe)
├── compute_metrics.py        Stats from CF JSONL outputs → metrics.json
├── visualize_cf.py           6 matplotlib figures (failure types, divergence, actions, trajectories)
├── approach_comparison.md    Design-decision table (static reference)
└── run_analysis.py           One-shot: serialize → metrics → visualize
```

```bash
cd CAST

# Bootstrap trajectories from atomic TFRecords (avoid downloading GoStanford2)
python analysis/reconstruct_episodes.py \
    --atomic-dir /home/vasu/thesis_work/atomic_datasets \
    --output-dir /path/to/reconstructed_trajectories \
    [--min-chunks 2] [--max-trajs 100]
# Then set dataset_path in data_gen.yaml to /path/to/reconstructed_trajectories

# After running the CF pipeline, generate all analysis outputs:
python analysis/run_analysis.py --config configs/data_gen.yaml
# Outputs: {output_dir}/analysis/metrics.json
#          {output_dir}/analysis/figures/*.png
#          {output_dir}/unsafe_action_generation/unsafe_action_generation_outputs.jsonl
```

**TFRecord structure** (from inspection of `atomic_forward_dataset`):
- `episode_metadata/file_path` — absolute path ending in `{base_traj_name}_chunk_{N}`. **This is the grouping key** (strip `_chunk_N`); `N` is the temporal sort key.
- `episode_metadata/episode_id` — per-shard integer counter, NOT shared across action classes.
- `steps/observation/image` — PNG bytes embedded directly (no external file dependency).
- `steps/observation/position` — float list, shape `(T*2,)`, reshape to `(T, 2)`.
- `steps/observation/yaw` — float list, shape `(T,)`.
- `steps/language_instruction` — atomic label only (e.g. `"Go forward"`); hindsight step needed for rich NL.

---

## Atomic Datasets

Downloaded TFDS datasets are at `/home/vasu/thesis_work/atomic_datasets/`, one subdirectory per action class, each with a `1.0.0/` version folder containing 256 TFRecord shards. These are used to train the atomic policy (`train_atomic.py`). Point `data_config_path` in `data_gen.yaml` at the appropriate config and set `action_model.model_path` to the trained checkpoint.

---

## Key Config Fields (`configs/data_gen.yaml`)

```yaml
dataset_path: /path/to/gnm/dataset      # local root with traj_data.pkl files
output_dir: /path/to/output
model_version: gemini-2.5-pro           # Gemini model (Vertex)
project_id: your-gcp-project
gcp_project_id: your-gcp-project
gcp_region: us-central1
base_uri: your-gcs-bucket-name          # bucket name only, no gs:// prefix
blob_root: cast
# Atomic decomposition thresholds:
min_turn_thres: 0.9
max_adjust_thres: 0.5
min_adjust_thres: 0.1
min_forward_thres: 0.5
max_atomic_chunk_length: 10
# Prompt/schema paths:
cf_step_prompt: /path/to/cf_step_prompt.txt
cf_step_schema: /path/to/cf_step_schema.json
```
