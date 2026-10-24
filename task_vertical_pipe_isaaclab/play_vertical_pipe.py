"""
==============================================================================
play_vertical_pipe.py - Watch / evaluate a trained policy in Isaac Sim
(port of D:\\mujoco\\Continuum_MuJoCo\\task_vertical_pipe\\demo_vertical_pipe.py)
==============================================================================
Examples (Isaac Lab python, from this folder):
    python play_vertical_pipe.py                                   # Isaac Sim window, rig + random pipe
    python play_vertical_pipe.py --pipe-mode rig --episodes 3
    python play_vertical_pipe.py --num_envs 10                     # 10 robots at once (5 rig, 5 random)
    python play_vertical_pipe.py --headless --episodes 100         # success statistics, no rendering
    python play_vertical_pipe.py --video --episodes 5              # MP4 with HUD in videos/
    python play_vertical_pipe.py --checkpoint models/ppo_vpipe/final_model.pt

The policy is the deterministic mean of the rsl_rl actor (like model.predict(deterministic=True)).
The window / video use the CAD meshes (assets/generated/usd/continuum_visual); statistics use the
collider-only USD. Physics is the same in both.
==============================================================================
"""

import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # Kit exits with os._exit: keep prints when stdout is a file

import numpy as np
import torch
import tensordict  # noqa: F401  (Windows: load DLLs before Kit)
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Play a trained PPO policy: continuum robot through a vertical pipe")
parser.add_argument("--checkpoint", type=str, default="", help="rsl_rl .pt (default: models/ppo_vpipe/best_model.pt)")
parser.add_argument("--pipe-mode", choices=["rig", "random", "both"], default="both")
parser.add_argument("--episodes", type=int, default=5, help="episodes per pipe mode")
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--max-offset", type=float, default=0.17, help="random scene: max pipe offset (m)")
parser.add_argument("--max-steps", type=int, default=400)
parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=0,
                    help="robots in the scene (window / video: all run at once, the camera shows the whole grid; "
                         "--headless: total envs for the statistics). Env i gets pipe mode i % #modes")
parser.add_argument("--env-spacing", type=float, default=2.0, help="distance between the robots (m)")
parser.add_argument("--stats-envs", type=int, default=16, help="--headless without --num_envs: parallel envs per pipe mode")
parser.add_argument("--video", action="store_true", help="record an MP4 (headless, needs cameras)")
parser.add_argument("--output", type=str, default="", help="MP4 path (default videos/vertical_pipe_<mode>.mp4)")
parser.add_argument("--slowdown", type=float, default=1.0, help=">1 plays slower than real time")
parser.add_argument("--hold", type=float, default=1.2, help="seconds to hold the final pose of each episode")
parser.add_argument("--no-meshes", action="store_true", help="show the collision cylinders instead of the CAD meshes")
parser.add_argument("--actuation", choices=["joint", "cable"], default="joint",
                    help="cable: tendon-driven robot (cable_env.py); the checkpoint must be trained with --actuation cable "
                         "(default checkpoint models/ppo_vpipe_cable/best_model.pt)")
parser.add_argument("--obs-tension", action="store_true", help="cable: the checkpoint observes the cable tensions (39-dim obs)")
parser.add_argument("--draw-cables", action="store_true", help="cable: draw the 12 cables (brightness = tension)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
if args.video:
    args.enable_cameras = True
    args.headless = True
app = AppLauncher(args).app

from tensordict import TensorDict  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402

import constants as C  # noqa: E402
from agent_cfg import make_agent_cfg  # noqa: E402
from pipe_runner import OUTCOMES  # noqa: E402
from cable_env import CableVerticalPipeEnv, CableVerticalPipeEnvCfg  # noqa: E402
from vertical_pipe_env import VerticalPipeEnv, VerticalPipeEnvCfg  # noqa: E402

FAILURE_TEXT = {"rim_hit": "HIT THE PIPE RIM", "missed_pipe": "MISSED THE PIPE", "unstable": "SIM UNSTABLE",
                "timeout": "TIME OUT"}


def find_checkpoint(path):
    if path:
        for cand in (path, os.path.join(TASK_DIR, path), os.path.join(TASK_DIR, "models", path)):
            if os.path.isfile(cand):
                return cand
        return ""
    run = "ppo_vpipe_cable" if args.actuation == "cable" else "ppo_vpipe"
    for run, name in ((run, "best_model.pt"), (run, "final_model.pt")):
        cand = os.path.join(TASK_DIR, "models", run, name)
        if os.path.isfile(cand):
            return cand
    return ""


def load_actor(path):
    """Deterministic rsl_rl actor with the architecture of agent_cfg.py."""
    actor_cfg = dict(make_agent_cfg(1, 1)["actor"])
    actor_cfg.pop("class_name")
    actor_cfg["distribution_cfg"] = dict(actor_cfg["distribution_cfg"])
    obs_dim = 33 + (6 if args.actuation == "cable" and args.obs_tension else 0)
    dummy = TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1])
    actor = MLPModel(dummy, {"actor": ["policy"], "critic": ["policy"]}, "actor", 7, **actor_cfg)
    state = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(state["actor_state_dict"])
    actor.eval()
    return actor


def make_env(modes, num_envs, visual, render_mode=None):
    cfg = CableVerticalPipeEnvCfg() if args.actuation == "cable" else VerticalPipeEnvCfg()
    if args.actuation == "cable":
        cfg.obs_tension = args.obs_tension
        cfg.draw_cables = args.draw_cables
    cfg.scene.num_envs = num_envs
    cfg.pipe_modes = tuple(modes)
    cfg.max_episode_steps = args.max_steps
    cfg.random_offset = (0.0, args.max_offset)
    cfg.assist_prob = 0.0
    cfg.seed = args.seed
    cfg.visual = visual
    cfg.scene.env_spacing = args.env_spacing
    cfg.sim.device = args.device
    if args.video:
        cfg.viewer.resolution = (1280, 720)
    return (CableVerticalPipeEnv if args.actuation == "cable" else VerticalPipeEnv)(cfg, render_mode=render_mode)


def act(actor, obs, env):
    td = TensorDict({"policy": obs["policy"].to("cpu")}, batch_size=[env.num_envs])
    return actor(td).to(env.device)


# ---------------------------------------------------------------------------
def run_stats(actor, modes):
    """Success statistics over many episodes (demo_vertical_pipe.py --headless)."""
    env = make_env(modes, max(args.num_envs, len(modes)) if args.num_envs else args.stats_envs * len(modes),
                   visual=False)
    n = env.num_envs
    results = {m: [] for m in modes}
    min_clear = np.full(n, np.inf)
    obs, _ = env.reset()
    while any(len(results[m]) < args.episodes for m in modes):
        obs, _, term, trunc, extras = env.step(act(actor, obs, env))
        info = extras["done_info"]
        min_clear = np.minimum(min_clear, info["min_clearance"].cpu().numpy())
        outcome = info["outcome"].cpu().numpy()
        for i in np.nonzero(outcome >= 0)[0]:
            mode = env.env_modes[i]
            if len(results[mode]) < args.episodes:
                results[mode].append((OUTCOMES[outcome[i]], int(info["contact_steps"][i]),
                                      int(info["episode_length"][i]), float(min_clear[i])))
            min_clear[i] = np.inf
    for mode in modes:
        res = results[mode]
        ok = [r for r in res if r[0] == "success"]
        fails = {}
        for r in res:
            if r[0] != "success":
                fails[r[0]] = fails.get(r[0], 0) + 1
        print(f"\n[{mode}] {len(res)} episodes: success {100 * len(ok) / len(res):.1f}%  "
              + "  ".join(f"{k}: {v}" for k, v in sorted(fails.items())))
        if ok:
            steps = np.mean([r[2] for r in ok])
            clear = np.array([r[3] for r in ok]) * 1000.0
            print(f"  successful episodes: {steps:.0f} steps ({steps * 0.02:.2f} s) on average, "
                  f"min wall clearance median {np.median(clear):.1f} mm (worst {clear.min():.1f} mm)")
        contact = np.array([r[1] for r in res])
        print(f"  wall-contact steps per episode: mean {contact.mean():.2f}, "
              f"episodes without any contact: {100 * np.mean(contact == 0):.0f}%")
    env.close()


# ---------------------------------------------------------------------------
def draw_hud(cv2, frame, env, i, ep, n_eps, step, successes, finished_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    info = env.info(i)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (560, 230), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    def put(text, y, color=(230, 230, 230), scale=0.45, thick=1):
        cv2.putText(frame, text, (20, y), font, scale, color, thick, cv2.LINE_AA)

    put("PPO policy (Isaac Sim): continuum robot through a vertical pipe", 32, (0, 230, 255))
    put(f"[{env.env_modes[i]}] episode {ep}/{n_eps}  step {step:3d}  t = {step * 0.02:4.2f} s  success {successes}", 54)
    put(f"Stage: {info['stage_name']}", 78, (0, 200, 255), 0.55, 2)
    put(f"Tip offset from pipe axis: {info['lat_mm']:5.1f} mm   tilt {info['tilt_deg']:4.1f} deg", 100)
    bore_mm = float(env.z_top[i] - env.z_success[i]) * 1000.0
    put(f"Depth: {max(info['depth_mm'], 0.0):5.1f} / {bore_mm:.0f} mm", 122)
    clear = f"{info['clearance_mm']:4.1f} mm" if np.isfinite(info["clearance_mm"]) else "  --"
    put(f"Wall clearance: {clear}   wall-contact steps: {info['contact_steps']}", 144)
    put(f"Elevator: {info['elevator_mm']:+6.1f} mm  (motor {info['elevator_motor_deg']:+7.0f} deg)", 166)
    m = info["motor_angles"]
    put("Lead-screw motors (deg): " + " ".join(f"{v:+5.0f}" for v in m), 190, (170, 200, 240))
    if finished_text:
        (tw, th), _ = cv2.getTextSize(finished_text, font, 0.9, 2)
        x, y = (frame.shape[1] - tw) // 2, frame.shape[0] - 60
        color = (40, 170, 40) if finished_text.startswith("THROUGH") else (40, 40, 200)
        cv2.rectangle(frame, (x - 20, y - th - 14), (x + tw + 20, y + 14), color, -1)
        cv2.putText(frame, finished_text, (x, y), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def run_live(actor, modes):
    """
    One env per pipe mode in the same scene (Isaac Sim runs one simulation per process). The scenes are
    shown one after the other like the MuJoCo demo: the camera (window or MP4) follows env k while its
    episodes are counted; the other env keeps running the policy in the background.
    """
    render_mode = "rgb_array" if args.video else None
    env = make_env(modes, len(modes), visual=not args.no_meshes, render_mode=render_mode)
    writer, out = None, ""
    if args.video:
        import cv2
        out = args.output or os.path.join(TASK_DIR, "videos", f"vertical_pipe_{args.pipe_mode}.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 50.0 / args.slowdown, (1280, 720))
    results = {}
    env.auto_reset = False                      # keep the final pose on screen, then reset explicitly
    obs, _ = env.reset()
    for k, mode in enumerate(modes):
        if env.viewport_camera_controller is not None:
            env.viewport_camera_controller.set_view_env_index(k)
        successes = 0
        for ep in range(1, args.episodes + 1):
            print(f"\n--- [{mode}] episode {ep}/{args.episodes} | pipe top at "
                  f"{np.round([*env.pipe_xy[k].cpu().numpy(), float(env.z_top[k])], 3)} m ---")
            step, done, finished = 0, False, ""
            while not done:
                t0 = time.time()
                obs, _, term, trunc, extras = env.step(act(actor, obs, env))
                step += 1
                info_k = extras["done_info"]
                outcome = int(info_k["outcome"][k])
                done = outcome >= 0
                info = env.info(k)
                if done:
                    name = OUTCOMES[outcome]
                    successes += int(name == "success")
                    finished = "THROUGH THE PIPE!" if name == "success" else FAILURE_TEXT.get(name, name.upper())
                    results.setdefault(mode, []).append(name)
                    print(f"\n  -> {name.upper()} after {step} steps | wall-contact steps "
                          f"{int(info_k['contact_steps'][k])} | max depth {float(info_k['max_depth'][k]) * 1000:5.1f} mm")
                else:
                    sys.stdout.write(f"\rstep {step:3d} [{info['stage_name']:6s}] offset {info['lat_mm']:5.1f} mm | "
                                     f"tilt {info['tilt_deg']:4.1f} deg | depth {info['depth_mm']:+6.1f} mm | "
                                     f"contacts {info['contact_steps']:3d} | elevator {info['elevator_mm']:+6.1f} mm ")
                if writer is not None:
                    import cv2
                    frame = cv2.cvtColor(np.ascontiguousarray(env.render()), cv2.COLOR_RGB2BGR)
                    frame = draw_hud(cv2, frame, env, k, ep, args.episodes, step, successes, finished)
                    for _ in range(int(args.hold * 50 / args.slowdown) if done else 1):
                        writer.write(frame)
                elif not args.headless:
                    time.sleep(max(0.0, 0.02 * args.slowdown - (time.time() - t0)))
                    if done:                        # hold the final pose
                        t_end = time.time() + args.hold
                        while time.time() < t_end and app.is_running():
                            env.sim.render()
                ended = [i for i, o in enumerate(info_k["outcome"].cpu().tolist()) if o >= 0]
                if ended:
                    obs = env.reset_envs(ended)
                if not app.is_running():
                    return
        print(f"\n[{mode}] success {successes}/{args.episodes}")
    if writer is not None:
        writer.release()
        print(f"video: {os.path.abspath(out)}")
    print("Summary: " + ", ".join(f"{m}: {r.count('success')}/{len(r)}" for m, r in results.items()))
    env.close()


def run_many(actor, modes, n):
    """
    --num_envs larger than the number of pipe modes: all robots run the policy at the same time (env i on
    pipe mode i % #modes) and the camera looks at the whole grid. Stops when every pipe mode has
    --episodes finished episodes.
    """
    render_mode = "rgb_array" if args.video else None
    env = make_env(modes, n, visual=not args.no_meshes, render_mode=render_mode)
    if env.viewport_camera_controller is not None:
        origins = env.scene.env_origins.cpu().numpy()
        centre = origins.mean(axis=0) + np.array([0.09, 0.21, 0.55])        # middle of the robot / pipe area
        extent = float(np.abs(origins[:, :2] - origins[:, :2].mean(axis=0)).max()) + 1.0
        env.viewport_camera_controller.update_view_to_world()
        env.viewport_camera_controller.update_view_location(
            eye=centre + np.array([1.3 * extent, -1.3 * extent, 0.9 * extent]), lookat=centre)
    writer, out = None, ""
    if args.video:
        import cv2
        out = args.output or os.path.join(TASK_DIR, "videos", f"vertical_pipe_{args.pipe_mode}_{n}envs.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 50.0 / args.slowdown, (1280, 720))
    results = {m: [] for m in modes}
    steps = np.zeros(n, dtype=int)
    env.auto_reset = False
    obs, _ = env.reset()
    while any(len(results[m]) < args.episodes for m in modes):
        t0 = time.time()
        obs, _, _, _, extras = env.step(act(actor, obs, env))
        steps += 1
        info = extras["done_info"]
        outcome = info["outcome"].cpu().numpy()
        ended = np.nonzero(outcome >= 0)[0].tolist()
        for i in ended:
            mode, name = env.env_modes[i], OUTCOMES[outcome[i]]
            if len(results[mode]) < args.episodes:
                results[mode].append(name)
            print(f"[env {i:2d} {mode:6s}] {name.upper():12s} after {steps[i]:3d} steps | "
                  f"wall-contact steps {int(info['contact_steps'][i])}")
            steps[i] = 0
        if writer is not None:
            import cv2
            frame = cv2.cvtColor(np.ascontiguousarray(env.render()), cv2.COLOR_RGB2BGR)
            text = f"{n} robots | " + " | ".join(
                f"{m}: {results[m].count('success')}/{len(results[m])} success" for m in modes)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
            writer.write(frame)
        elif not args.headless:
            time.sleep(max(0.0, 0.02 * args.slowdown - (time.time() - t0)))
        if ended:
            obs = env.reset_envs(ended)
        if not app.is_running():
            return
    if writer is not None:
        writer.release()
        print(f"video: {os.path.abspath(out)}")
    print("Summary: " + ", ".join(f"{m}: {r.count('success')}/{len(r)}" for m, r in results.items()))
    env.close()


def main():
    path = find_checkpoint(args.checkpoint)
    if not path:
        print("No trained model found. Train first:  python train_vertical_pipe.py")
        return
    print(f"Model: {os.path.abspath(path)} | scene: {args.pipe_mode} | PhysX {args.device}")
    actor = load_actor(path)
    modes = ["rig", "random"] if args.pipe_mode == "both" else [args.pipe_mode]
    with torch.inference_mode():
        if args.headless and not args.video:
            run_stats(actor, modes)
        elif args.num_envs > len(modes):
            run_many(actor, modes, args.num_envs)
        else:
            run_live(actor, modes)


if __name__ == "__main__":
    main()
    app.close()
