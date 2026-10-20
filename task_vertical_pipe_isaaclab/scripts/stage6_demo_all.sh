#!/usr/bin/env bash
# Stage 6 demos after the benchmarks: shape tracking (ND2), the language pipeline on uncertain robots with and
# without the tension CBF, the before / after clips. The language runs need the VLM in its own process:
#   python vlm.py --serve --device cpu
#   bash scripts/stage6_demo_all.sh            (from task_vertical_pipe_isaaclab/, Git Bash)
set -u
cd "$(dirname "$0")/.."
PY=/d/Isaacsim/env_isaacsim62/Scripts/python.exe
VLM=http://127.0.0.1:8765/ground

step() {   # name, command...
  echo "=== $(date +%H:%M:%S) start $1"
  "${@:2}" > "logs/s8_$1.log" 2>&1
  echo "=== $(date +%H:%M:%S) end $1 (exit $?)"
}

step shape_t525 $PY play_shape.py --pretension 5.25 --levels nominal,train --trajectories scurve,circle --robots 32 \
  --payload-step --cbf-source loadcell_id --tag shape_t525
step language_t525 $PY play_language.py --cable --robust train --pretension 5.25 --episodes 48 --vlm-url $VLM \
  --json reports/language/language_t525.json
step language_t525_safe $PY play_language.py --cable --robust train --pretension 5.25 --safe loadcell_id --episodes 48 \
  --vlm-url $VLM --json reports/language/language_t525_safe.json
step video_before $PY record_safety_video.py --policy models/ppo_cable_128/best_model.pt \
  --label "PPO of the cable stage" --pretension 3 --level train --episodes 3 --output videos/safety_before.mp4
step video_after $PY record_safety_video.py --policy models/lag_t525/best_safe_model.pt \
  --label "PPO-Lagrangian + CBF" --pretension 5.25 --cbf --cbf-source loadcell_id --level train \
  --episodes 3 --output videos/safety_after.mp4
echo "=== all done"
