#!/usr/bin/env bash
# Stage 6 training runs, one after the other (one Isaac Sim at a time on the 8 GB GPU).
# Every run fine-tunes the motor policy of the cable stage (models/ppo_cable_128) on robots of the "train"
# uncertainty set: 512 envs x 32 steps per update, 2M steps, evaluation every 200k steps on 128 episodes.
#   bash scripts/stage6_train_all.sh            (from task_vertical_pipe_isaaclab/, Git Bash)
set -u
cd "$(dirname "$0")/.."
PY=/d/Isaacsim/env_isaacsim62/Scripts/python.exe
COMMON="--robust train --resume models/ppo_cable_128/best_model.pt --n-envs 512 --steps-per-env 32 \
 --timesteps 2000000 --eval-freq 200000 --eval-episodes 128 --checkpoint-freq 500000"

run() {   # run-name, extra args
  echo "=== $(date +%H:%M:%S) start $1"
  $PY train_vertical_pipe.py $COMMON --run-name "$1" "${@:2}" > "logs/s6_train_$1.log" 2>&1
  echo "=== $(date +%H:%M:%S) end $1 (exit $?)"
  grep -E "Eval num|Done in|Traceback" "logs/s6_train_$1.log" | tail -4
}

run lag_t3      --constrained --priv critic --pretension 3
run lag_t525    --constrained --priv critic --pretension 5.25
run dr_t525     --priv critic --pretension 5.25
run dr_t3       --priv critic --pretension 3
run teacher_t525 --constrained --priv both --pretension 5.25
run lag_t525_lc --constrained --priv critic --obs-tension --pretension 5.25

echo "=== $(date +%H:%M:%S) start rma"
$PY train_rma.py --teacher models/teacher_t525/best_safe_model.pt --pretension 5.25 --num-envs 256 \
  --iterations 60 --run-name rma_t525 > logs/s6_train_rma_t525.log 2>&1
echo "=== $(date +%H:%M:%S) end rma (exit $?)"
tail -3 logs/s6_train_rma_t525.log
echo "=== all done"
