#!/usr/bin/env bash
# K22-SafeBench of Stage 6 (after scripts/stage6_train_all.sh): every controller on the same robots / pipes
# (seed 9000), 100 episodes per pipe mode and cell, levels nominal / train / ood, then the stress tests on
# the train set. One process per pre-tension (the env's T0 is fixed at creation).
#   bash scripts/stage6_bench_all.sh            (from task_vertical_pipe_isaaclab/, Git Bash)
set -u
cd "$(dirname "$0")/.."
PY=/d/Isaacsim/env_isaacsim62/Scripts/python.exe

bench() {   # tag, pre-tension, extra args (policies, levels, stress)
  echo "=== $(date +%H:%M:%S) start $1"
  $PY benchmark_safety.py --tag "$1" --pretension "$2" "${@:3}" > "logs/s7_bench_$1.log" 2>&1
  echo "=== $(date +%H:%M:%S) end $1 (exit $?)"
  grep -E "Traceback|Error" "logs/s7_bench_$1.log" | tail -3
  $PY plot_safebench.py "reports/safebench/$1.json" > /dev/null 2>&1 || echo "plot failed: $1"
}

T3="--policy ppo_nominal=models/ppo_cable_128/best_model.pt \
 --policy ppo_nominal+cbf=models/ppo_cable_128/best_model.pt+cbf \
 --policy dr_t3=models/dr_t3/best_model.pt \
 --policy lag_t3=models/lag_t3/best_safe_model.pt \
 --policy lag_t3+cbf=models/lag_t3/best_safe_model.pt+cbf"

T525="--policy ppo_nominal=models/ppo_cable_128/best_model.pt \
 --policy ppo_nominal+cbf=models/ppo_cable_128/best_model.pt+cbf \
 --policy dr_t525=models/dr_t525/best_model.pt \
 --policy lag_t525=models/lag_t525/best_safe_model.pt \
 --policy lag_t525+cbf=models/lag_t525/best_safe_model.pt+cbf \
 --policy lag_t525+cbf_model=models/lag_t525/best_safe_model.pt+cbf-model \
 --policy lag_t525+cbf_oracle=models/lag_t525/best_safe_model.pt+cbf-oracle \
 --policy lag_t525_lc=models/lag_t525_lc/best_safe_model.pt \
 --policy teacher_t525=models/teacher_t525/best_safe_model.pt \
 --policy rma_t525=models/rma_t525/best.pt \
 --policy rma_t525+cbf=models/rma_t525/best.pt+cbf"

bench stage6_t3   3    $T3   --levels nominal,train,ood
bench stage6_t525 5.25 $T525 --levels nominal,train,ood
bench stress_t3   3    $T3   --levels train --stress push,creep,backlash
bench stress_t525 5.25 $T525 --levels train --stress push,creep,backlash
echo "=== all done"
