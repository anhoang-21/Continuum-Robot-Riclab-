"""
==============================================================================
train_student.py - DAgger: image-based student from the privileged PPO teacher
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe train_student.py
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe train_student.py --num_envs 32 --iterations 150 --meshes

Every iteration: all envs run for --steps-per-iter steps; each episode is driven by the
teacher with probability beta (1 in the first iteration, down to 0 after --beta-iters
iterations), otherwise by the student. Every visited state is stored with the teacher's
action (the teacher reads the true pipe pose, the student only the tip camera and the
joint commands); the student is then trained on the whole buffer (MSE, random-shift
augmentation). Starts: preinsert.StudentStartSampler (pipe in view, as after a search).

Printed per iteration: loss, success rate of the last student-driven and teacher-driven
episodes. Checkpoints: models/<run>/student_<it>.pt, best.pt (student success), final.pt.
==============================================================================
"""

import argparse
import os
import sys
import time
from collections import deque

sys.stdout.reconfigure(line_buffering=True)  # Kit exits with os._exit: keep prints when stdout is a file

import numpy as np
import torch
import tensordict  # noqa: F401  (Windows: load DLLs before Kit)
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="DAgger: tip-camera student from the privileged PPO teacher")
parser.add_argument("--teacher", type=str, default=os.path.join("models", "mujoco_ppo_vpipe_wide", "model.pt"))
parser.add_argument("--run-name", type=str, default="student_vpipe")
parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=32)
parser.add_argument("--iterations", type=int, default=150)
parser.add_argument("--steps-per-iter", type=int, default=64)
parser.add_argument("--beta-iters", type=int, default=10, help="iterations until only the student drives")
parser.add_argument("--grad-steps", type=int, default=150)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--buffer", type=int, default=250_000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-offset", type=float, default=0.17)
parser.add_argument("--meshes", action="store_true", help="CAD frame / plate / robot in the camera view")
parser.add_argument("--depth-noise", type=float, default=0.0)
parser.add_argument("--rgb-noise", type=float, default=0.0)
parser.add_argument("--save-every", type=int, default=25)
parser.add_argument("--multi", action="store_true",
                    help="Stage 5 scenes: 3 pipes with random tube colours around (multi_pipe_env.py)")
parser.add_argument("--train-tube-prob", type=float, default=0.25,
                    help="--multi: chance that the target keeps the light-blue tube of the single-pipe task")
parser.add_argument("--init", type=str, default="", help="start from this student checkpoint")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

from tensordict import TensorDict  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from pipe_runner import OUTCOMES  # noqa: E402
from student_env import StudentPipeEnv, StudentPipeEnvCfg  # noqa: E402
from student_policy import IMG, PROPRIO_DIM, StudentPolicy, load_student, save_student, student_inputs  # noqa: E402
from vision_pipeline import load_actor  # noqa: E402


class Buffer:
    """Ring buffer on the CPU (uint8 images)."""

    def __init__(self, size):
        self.img = torch.zeros(size, 4, IMG, IMG, dtype=torch.uint8)
        self.prop = torch.zeros(size, PROPRIO_DIM)
        self.act = torch.zeros(size, 7)
        self.size, self.n, self.pos = size, 0, 0

    def add(self, img, prop, act):
        k = img.shape[0]
        idx = (torch.arange(k) + self.pos) % self.size
        self.img[idx], self.prop[idx], self.act[idx] = img.cpu(), prop.cpu(), act.cpu()
        self.pos = (self.pos + k) % self.size
        self.n = min(self.n + k, self.size)

    def sample(self, k, device):
        idx = torch.randint(0, self.n, (k,))
        return self.img[idx].to(device), self.prop[idx].to(device), self.act[idx].to(device)


def main():
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.multi:
        from multi_pipe_env import MultiPipeEnv, MultiPipeEnvCfg

        cfg = MultiPipeEnvCfg()
        cfg.student_starts = True
        cfg.overview = False
        cfg.obs_source = "gt"
        cfg.max_episode_steps = 400
        cfg.train_tube_prob = args.train_tube_prob
    else:
        cfg = StudentPipeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.random_offset = (0.0, args.max_offset)
    cfg.assist_prob = 0.0
    cfg.visual = args.meshes
    cfg.depth_noise_std = args.depth_noise
    cfg.rgb_noise_std = args.rgb_noise
    cfg.sim.device = args.device
    env = MultiPipeEnv(cfg) if args.multi else StudentPipeEnv(cfg)
    dev, n = env.device, env.num_envs

    teacher = load_actor(os.path.join(TASK_DIR, args.teacher) if not os.path.isabs(args.teacher) else args.teacher)
    student = load_student(os.path.join(TASK_DIR, args.init) if args.init and not os.path.isabs(args.init) else args.init,
                           device=dev).train() if args.init else StudentPolicy().to(dev)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    buf = Buffer(args.buffer)
    out_dir = os.path.join(TASK_DIR, "models", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(TASK_DIR, "logs", args.run_name))

    stud_hist, teach_hist = deque(maxlen=200), deque(maxlen=200)
    by_teacher = np.ones(n, dtype=bool)
    stale = np.ones(n, dtype=bool)                  # camera frame still shows the pose before the reset
    obs, _ = env.reset()
    best, t_start, total = -1.0, time.time(), 0
    for it in range(args.iterations):
        beta = max(0.0, 1.0 - it / max(args.beta_iters, 1))
        t0 = time.time()
        student.eval()
        with torch.inference_mode():
            for _ in range(args.steps_per_iter):
                img, prop = student_inputs(env)
                a_t = teacher(TensorDict({"policy": obs["policy"].cpu()}, batch_size=[n])).clamp(-1.0, 1.0).to(dev)
                a_s = student.act(img, prop)
                use_t = torch.as_tensor(by_teacher, device=dev)[:, None]
                act = torch.where(use_t, a_t, a_s)
                fresh = torch.as_tensor(~stale, device=dev)
                act[~fresh] = 0.0                   # hold still until the camera shows the new start
                if fresh.any():
                    buf.add(img[fresh], prop[fresh], a_t[fresh])
                stale[:] = False
                obs, _, term, trunc, extras = env.step(act)
                total += n
                outcome = extras["done_info"]["outcome"].cpu().numpy()
                for i in np.nonzero(outcome >= 0)[0]:
                    (teach_hist if by_teacher[i] else stud_hist).append(OUTCOMES[outcome[i]] == "success")
                    by_teacher[i] = rng.uniform() < beta
                    stale[i] = True
        collect = time.time() - t0

        t0 = time.time()
        student.train()
        losses = []
        for _ in range(args.grad_steps if buf.n >= args.batch else 0):
            img, prop, a = buf.sample(args.batch, dev)
            loss = torch.nn.functional.mse_loss(student(img, prop, augment=True), a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train = time.time() - t0

        s_rate = float(np.mean(stud_hist)) if stud_hist else float("nan")
        t_rate = float(np.mean(teach_hist)) if teach_hist else float("nan")
        loss = float(np.mean(losses)) if losses else float("nan")
        writer.add_scalar("student/loss", loss, total)
        writer.add_scalar("student/beta", beta, total)
        if stud_hist:
            writer.add_scalar("student/success_rate", s_rate, total)
        if teach_hist:
            writer.add_scalar("teacher/success_rate", t_rate, total)
        print(f"it {it:4d} | beta {beta:4.2f} | loss {loss:.4f} | buffer {buf.n:7d} | student success "
              f"{s_rate * 100:5.1f}% ({len(stud_hist)} eps) | teacher {t_rate * 100:5.1f}% ({len(teach_hist)} eps) | "
              f"{args.steps_per_iter * n / collect:5.0f} steps/s, train {train:4.1f} s", flush=True)
        if beta == 0.0 and len(stud_hist) >= 100 and s_rate > best:
            best = s_rate
            save_student(student, os.path.join(out_dir, "best.pt"), success=s_rate, iteration=it, steps=total)
        if (it + 1) % args.save_every == 0:
            save_student(student, os.path.join(out_dir, f"student_{it + 1}.pt"), iteration=it, steps=total)
    save_student(student, os.path.join(out_dir, "final.pt"), iteration=args.iterations, steps=total)
    print(f"done: {total:,} env steps in {(time.time() - t_start) / 60:.1f} min, best student success "
          f"{best * 100:.1f}% -> {out_dir}")
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
