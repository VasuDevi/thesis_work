#!/usr/bin/env bash
# setup_github.sh — Create GitHub repos and push everything
#
# Usage: GITHUB_TOKEN=ghp_xxxx bash setup_github.sh
#
# Creates:
#   1. VasuDevi/CAST    — fork of the CAST modifications
#   2. VasuDevi/thesis_work — umbrella repo (submodules + nwm code + configs)
#
# Requirements: GITHUB_TOKEN env var with 'repo' scope

set -euo pipefail

TOKEN="${GITHUB_TOKEN:?Set GITHUB_TOKEN=ghp_xxx}"
GITHUB_USER="VasuDevi"
API="https://api.github.com"

_create_repo() {
  local name="$1" desc="$2"
  echo "Creating github.com/$GITHUB_USER/$name ..."
  curl -s -X POST "$API/user/repos" \
    -H "Authorization: token $TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    -d "{\"name\":\"$name\",\"description\":\"$desc\",\"private\":false}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('html_url', r.get('message','?')))"
}

# ── 1. Push CAST fork to VasuDevi/CAST ─────────────────────────────────────
echo ""
echo "=== Setting up VasuDevi/CAST ==="
_create_repo "CAST" "Safety-critical CAST pipeline with unsafe counterfactual generation, NWM rollout, and DaCBaF safety policy"

cd /home/vasu/thesis_work/CAST
git remote add vasu "git@github.com:$GITHUB_USER/CAST.git" 2>/dev/null || true
git push vasu main

# ── 2. Init thesis_work umbrella repo ──────────────────────────────────────
echo ""
echo "=== Setting up VasuDevi/thesis_work ==="
_create_repo "thesis_work" "Unified thesis repository: CAST safety pipeline, Neural CBF, NWM, DROID dataset generation"

cd /home/vasu/thesis_work

# Already initialized as git repo in earlier step (or re-init)
git init 2>/dev/null || true
git branch -M main 2>/dev/null || true

# Add submodules pointing to VasuDevi's repos
git submodule add "git@github.com:$GITHUB_USER/thesis.git" thesis 2>/dev/null || true
git submodule add "git@github.com:$GITHUB_USER/CAST.git" CAST 2>/dev/null || true

# Commit submodules + other code
git add .gitignore CLAUDE.md .gitmodules nwm/ setup_github.sh setup_envs.sh 2>/dev/null || true
git add .gitmodules 2>/dev/null || true
git commit -m "$(cat <<'EOF'
Initial thesis_work umbrella repo

Submodules:
  thesis/ → VasuDevi/thesis (NCF, DaCBaF, DROID pipeline)
  CAST/   → VasuDevi/CAST (unsafe CF, NWM rollout, safety margins)

Direct code:
  nwm/    → NWM (CDiT-XL/2) inference code (no large model weights)

Large files (datasets/checkpoints) → see HuggingFace repos below:
  Checkpoints:   https://huggingface.co/Sswa12/thesis-ncbf-checkpoints
  Outputs/data:  https://huggingface.co/datasets/Sswa12/thesis-cast-outputs

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)" 2>/dev/null || true

git remote add origin "git@github.com:$GITHUB_USER/thesis_work.git" 2>/dev/null || true
git push -u origin main

echo ""
echo "Done! Repos available at:"
echo "  https://github.com/$GITHUB_USER/thesis"
echo "  https://github.com/$GITHUB_USER/CAST"
echo "  https://github.com/$GITHUB_USER/thesis_work"
