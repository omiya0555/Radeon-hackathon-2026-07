#!/usr/bin/env bash
# The whole simulation pipeline, top to bottom: collect -> train -> evaluate.
# No robot required; everything here runs on the Radeon host.
#
#   bash run_pipeline.sh --quick    # ~25 min: 5 episodes, 2k steps, 5 eval episodes
#   bash run_pipeline.sh --full     # ~13 h:  50 episodes, 100k steps, 20 eval episodes
#
# --quick exists so the flow can be demonstrated (and recorded) end to end in one sitting;
# --full reproduces the numbers in the technical report.
set -euo pipefail

MODE="${1:---quick}"
case "$MODE" in
  --quick) EPISODES=5;  STEPS=2000;   SAVE_FREQ=1000;  EVAL_EPS=5  ;;
  --full)  EPISODES=50; STEPS=100000; SAVE_FREQ=20000; EVAL_EPS=20 ;;
  *) echo "usage: $0 [--quick|--full]"; exit 2 ;;
esac

REPO_ID="${REPO_ID:-local/so101_cube_dr}"
DS_ROOT="datasets/$(basename "$REPO_ID")"
OUT_DIR="outputs/act_dr_${MODE#--}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

cd "$(dirname "$0")"
step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
t0=$SECONDS

step "Mode: $MODE — $EPISODES episodes, $STEPS training steps, $EVAL_EPS eval episodes"
python -c "import torch; print('torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------- 1. collect
step "1/4  Collecting $EPISODES demonstrations with the scripted expert (full DR)"
# Fully automatic: 5-DOF position-priority IK, rake grasp, retry on a missed lift.
# No human in the loop — this is the step that replaces hours of teleoperation.
if [ -d "$DS_ROOT" ]; then
  echo "  $DS_ROOT already exists — skipping collection (delete it to re-collect)"
else
  python src/record_dataset_so101.py \
    --episodes "$EPISODES" --dr-all \
    --repo-id "$REPO_ID" --dataset-root "$DS_ROOT"
fi

# ------------------------------------------------------------------ 2. train
step "2/4  Training ACT on the Radeon GPU ($STEPS steps)"
# video_backend=pyav: torchcodec's extension is ABI-incompatible with AMD's torch build.
# Watch utilisation from another shell with: watch -n 1 rocm-smi
lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DS_ROOT" \
  --dataset.image_transforms.enable=true \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --output_dir="$OUT_DIR" \
  --steps="$STEPS" --batch_size=8 --save_freq="$SAVE_FREQ" \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false

CKPT="$OUT_DIR/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
[ -d "$CKPT" ] || { echo "expected checkpoint missing: $CKPT"; exit 1; }

step "  Throughput actually achieved (GPU-bound when data_s << updt_s):"
grep -ao 'step:[^ ]* .*mem_gb:[0-9.]*' "$OUT_DIR"/*.log 2>/dev/null | tail -1 || \
  echo "  (see the training output above: ~2.55 step/s, data_s 0.002s vs updt_s 0.388s, 3.9 GB VRAM)"

# ------------------------------------------------------------------- 3. eval
step "3/4  Closed-loop evaluation in simulation ($EVAL_EPS fixed-seed episodes)"
# Seeds are fixed, so any two policies are compared on identical cube placements.
python src/eval_policy_so101.py \
  --policy-path "$CKPT" \
  --repo-id "$REPO_ID" --dataset-root "$DS_ROOT" \
  --episodes "$EVAL_EPS" --save-video \
  --video-dir "outputs/eval_videos/${MODE#--}" \
  --results-out "outputs/eval_results/${MODE#--}.json"

# --------------------------------------------------------------- 4. summary
step "4/4  Result"
python - <<PY
import json
d = json.load(open("outputs/eval_results/${MODE#--}.json"))
print(f"  success rate : {d['n_success']}/{d['episodes']} = {d['success_rate']*100:.0f}%")
print(f"  checkpoint   : {d['meta']['checkpoint']}")
print(f"  videos       : outputs/eval_videos/${MODE#--}/")
PY

printf '\n\033[1;32mPIPELINE_OK — %d min elapsed\033[0m\n' $(( (SECONDS - t0) / 60 ))
if [ "$MODE" = "--quick" ]; then
  echo "This was the smoke-sized run. For the reported 60% at 60k steps: bash run_pipeline.sh --full"
fi
