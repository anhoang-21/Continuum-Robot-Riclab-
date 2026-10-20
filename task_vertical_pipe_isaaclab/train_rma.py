"""
==============================================================================
train_rma.py - RMA phase 2: the adaptation module that estimates the robot's parameters online
==============================================================================
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe train_rma.py --teacher models/ppo_teacher/best_model.pt

The teacher (train_vertical_pipe.py --robust train --priv both) reads the robot's normalised parameters e.
The adaptation module phi (rma.AdaptationModule) learns e_hat = phi(last 50 observations) by regression on
the states of DAgger rollouts: each episode is driven by the teacher with the true e with probability beta
(1 -> 0 over --beta-iters iterations), otherwise by the deployed student pi(o, phi(history)). Every 4th
state is stored with (history, e) in a ring buffer; phi is trained on the buffer (MSE on e scaled per
parameter). Robots from robust_env.RobustCableEnv (--level, new parameters every episode).

Printed per iteration: loss, R^2 of e_hat per parameter on the last rollouts, success of student- and
teacher-driven episodes. Checkpoints: models/<run>/rma_<it>.pt, best.pt (student success, then loss), final.pt;
benchmark_safety.py / controllers.load_controller load them.
==============================================================================
"""

import argparse
import os
import sys
import time
from collections import deque

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import tensordict  # noqa: F401
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

from app_launch import launcher_args  # noqa: E402

parser = argparse.ArgumentParser(description="RMA phase 2: adaptation module (online parameter estimation)")
parser.add_argument("--teacher", type=str, required=True, help="PPO actor with the privileged parameters (--priv both)")
parser.add_argument("--run-name", type=str, default="rma")
parser.add_argument("--level", choices=["train", "ood"], default="train")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=60)
parser.add_argument("--steps-per-iter", type=int, default=100)
parser.add_argument("--beta-iters", type=int, default=8)
parser.add_argument("--store-every", type=int, default=4)
parser.add_argument("--buffer", type=int, default=60_000)
parser.add_argument("--grad-steps", type=int, default=200)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--assist", type=float, default=0.4)
parser.add_argument("--pretension", type=float, default=None, help="nominal cable pre-tension T0 (N), as the teacher's")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
app = AppLauncher(launcher_args(args, cameras=False)).app

from torch.utils.tensorboard import SummaryWriter  # noqa: E402

import uncertainty as U  # noqa: E402
from controllers import MLPPolicy  # noqa: E402
from pipe_runner import OUTCOMES  # noqa: E402
from rma import HISTORY, AdaptationModule, RMAPolicy  # noqa: E402
from robust_env import RobustCableEnv, RobustCableEnvCfg  # noqa: E402


def main():
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    teacher_path = os.path.abspath(args.teacher if os.path.isabs(args.teacher) else os.path.join(TASK_DIR, args.teacher))
    teacher = MLPPolicy(teacher_path, device=args.device)
    if not teacher.uses_priv:
        raise SystemExit(f"{args.teacher} does not read the privileged parameters (train it with --priv both)")
    models_dir = os.path.join(TASK_DIR, "models", args.run_name)
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(TASK_DIR, "logs", args.run_name))

    cfg = RobustCableEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.uncertainty = args.level
    cfg.pipe_modes = ("rig", "random")
    cfg.assist_prob = args.assist
    cfg.obs_tension = teacher.uses_tension
    cfg.priv_obs = True
    cfg.seed = args.seed
    if args.pretension is not None:
        cfg.cable_pretension = args.pretension
    cfg.sim.device = args.device
    env = RobustCableEnv(cfg)
    n, dev = env.num_envs, env.device
    obs_dim = 33 + (12 if teacher.uses_tension else 0)
    adapter = AdaptationModule(obs_dim).to(dev)
    student = RMAPolicy(teacher, adapter, num_envs=n, device=dev, name="rma")
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    buf_h = torch.zeros(args.buffer, HISTORY, obs_dim, dtype=torch.float16, device=dev)
    buf_e = torch.zeros(args.buffer, U.PRIV_DIM, device=dev)
    size, ptr = 0, 0
    student_out, teacher_out = deque(maxlen=200), deque(maxlen=200)
    best = (-1.0, np.inf)
    obs, _ = env.reset()
    student.reset()
    by_teacher = torch.as_tensor(rng.uniform(size=n) < 1.0, device=dev)
    t0 = time.time()
    for it in range(args.iterations):
        beta = max(0.0, 1.0 - it / max(args.beta_iters, 1))
        adapter.eval()
        est, tru = [], []
        for k in range(args.steps_per_iter):
            student.push(obs["policy"])
            with torch.no_grad():
                e_hat = adapter(student.hist)
            e = obs["priv"].to(dev)
            x = obs["policy"][:, :obs_dim].to(dev)
            e_in = torch.where(by_teacher[:, None], e, e_hat)
            with torch.no_grad():
                act = teacher.net(torch.cat([x[:, : (45 if teacher.uses_tension else 33)], e_in], dim=1))
            if k % args.store_every == 0:
                m = min(n, args.buffer - ptr)
                buf_h[ptr:ptr + m] = student.hist[:m].half()
                buf_e[ptr:ptr + m] = e[:m]
                ptr = (ptr + m) % args.buffer
                size = min(size + m, args.buffer)
                est.append(e_hat.cpu())
                tru.append(e.cpu())
            obs, _, _, _, ex = env.step(act)
            oc = ex["done_info"]["outcome"].cpu().numpy()
            ended = np.nonzero(oc >= 0)[0]
            for i in ended:
                (teacher_out if bool(by_teacher[i]) else student_out).append(OUTCOMES[oc[i]] == "success")
            if len(ended):
                student.reset(ended.tolist())
                by_teacher[torch.as_tensor(ended, device=dev)] = torch.as_tensor(
                    rng.uniform(size=len(ended)) < beta, device=dev)
        # train phi on the buffer
        adapter.train()
        scale = buf_e[:size].std(0).clamp(min=1e-3)
        losses = []
        for _ in range(args.grad_steps):
            idx = torch.randint(0, size, (args.batch,), device=dev)
            pred = adapter(buf_h[idx].float())
            loss = (((pred - buf_e[idx]) / scale) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        est, tru = torch.cat(est), torch.cat(tru)
        r2 = 1.0 - ((est - tru) ** 2).mean(0) / tru.var(0).clamp(min=1e-6)
        s_rate = float(np.mean(student_out)) if student_out else float("nan")
        t_rate = float(np.mean(teacher_out)) if teacher_out else float("nan")
        writer.add_scalar("rma/loss", np.mean(losses), it)
        writer.add_scalar("rma/student_success", s_rate, it)
        for j, name in enumerate(U.PRIV_NAMES):
            writer.add_scalar(f"rma_r2/{name}", float(r2[j]), it)
        print(f"[it {it + 1:3d}] beta {beta:.2f} | loss {np.mean(losses):.3f} | success student {100 * s_rate:5.1f}% "
              f"({len(student_out)}), teacher {100 * t_rate:5.1f}% ({len(teacher_out)}) | buffer {size} | "
              f"R2 " + " ".join(f"{nm}:{float(v):.2f}" for nm, v in zip(U.PRIV_NAMES, r2)) + f" | {time.time() - t0:.0f} s",
              flush=True)
        student.save(os.path.join(models_dir, "final.pt"), teacher_path)
        if (it + 1) % 10 == 0:
            student.save(os.path.join(models_dir, f"rma_{it + 1}.pt"), teacher_path)
        if beta == 0.0 and len(student_out) >= 50 and (s_rate, -np.mean(losses)) > (best[0], -best[1]):
            best = (s_rate, np.mean(losses))
            student.save(os.path.join(models_dir, "best.pt"), teacher_path)
    print(f"done: {models_dir}  (best student success {100 * best[0]:.1f}%)")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
