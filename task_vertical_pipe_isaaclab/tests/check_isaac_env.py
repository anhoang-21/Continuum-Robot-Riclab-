"""
==============================================================================
check_isaac_env.py - Checks of the Isaac Lab env (step 6 of the port)
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe tests/check_isaac_env.py

Run tests/compare_with_mujoco.py first (MuJoCo venv); it writes the reference
trajectories replayed here.

1. USD kinematics: random joint states written to PhysX, link poses compared with
   kinematics.py (which is identical to MuJoCo, see compare_with_mujoco.py)
2. resets: every env's start state equals the MuJoCo sampler for seed + env index
3. replay: the MuJoCo reference episodes (start state + actions) are replayed in
   PhysX; observations, rewards, contacts and episode ends are compared step by step
4. random actions on all envs (assisted starts on): finite observations, outcomes, speed
5. snapshot / restore used for evaluation during training
==============================================================================
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)  # Kit exits with os._exit: keep prints when stdout is a file
import time

import numpy as np
import torch
import tensordict  # noqa: F401  (Windows: load DLLs before Kit)

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--reference", default=os.path.join(TASK_DIR, "tests", "data", "mujoco_reference.npz"))
parser.add_argument("--random-steps", type=int, default=400)
parser.add_argument("--touch-tol", type=float, default=None, help="override cfg.contact_touch_tolerance (m)")
parser.add_argument("--rim-tol", type=float, nargs="*", default=None,
                    help="override cfg.rim_touch_tolerance (m); several values = replay once per value")
parser.add_argument("--only-replay", action="store_true")
AppLauncher.add_app_launcher_args(parser)
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import constants as C  # noqa: E402
from reset_logic import ResetSampler  # noqa: E402
from vertical_pipe_env import SEG_NAMES, VerticalPipeEnv, VerticalPipeEnvCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

N_ENVS = int(len(np.load(args.reference)["lengths"])) if os.path.exists(args.reference) else 36


def check_kinematics(env, trials=10):
    kin = env.kin
    names = env.robot.body_names
    kin_ids = [kin.body_index[n] for n in names if n in kin.body_index]
    isaac_ids = [i for i, n in enumerate(names) if n in kin.body_index]
    rng = np.random.default_rng(1)
    worst_p = worst_r = 0.0
    for _ in range(trials):
        q = rng.uniform(kin.joint_range[:, 0], kin.joint_range[:, 1], size=(env.num_envs, kin.nq))
        joint_pos = torch.zeros(env.num_envs, env.robot.num_joints, device=env.device)
        joint_pos[:, env._q_to_joint] = torch.as_tensor(q, dtype=torch.float32, device=env.device)
        env.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        pos = (env.robot.data.body_link_pos_w - env.scene.env_origins[:, None]).cpu().double().numpy()
        rot = matrix_from_quat(env.robot.data.body_link_quat_w.double()).cpu().numpy()
        xpos, xmat = kin.forward(q)
        worst_p = max(worst_p, np.abs(pos[:, isaac_ids] - xpos[:, kin_ids]).max())
        worst_r = max(worst_r, np.abs(rot[:, isaac_ids] - xmat[:, kin_ids]).max())
    print(f"[1] USD kinematics vs MJCF: {trials}x{env.num_envs} random joint states, "
          f"max |dpos| = {worst_p * 1000:.4f} mm, max |dR| = {worst_r:.2e}")
    assert worst_p < 1e-4 and worst_r < 1e-4
    return worst_p


def check_resets(env):
    env.cfg.max_episode_steps = 400
    env.assist_prob = 0.4
    env._rngs = [np.random.default_rng(env.cfg.seed + i) for i in range(env.num_envs)]
    env.reset()
    worst = 0.0
    jp = env.robot.data.joint_pos.cpu().double().numpy()
    for i in range(env.num_envs):
        sampler = ResetSampler(env.kin, env.env_modes[i], env.cfg.pipe_height, env.cfg.random_offset)
        st = sampler.sample(np.random.default_rng(env.cfg.seed + i), 0.4)
        q = env.kin.qpos_from(st.bend, st.elev)
        worst = max(worst, np.abs(env.pipe_xy[i].cpu().numpy() - st.pipe_xy).max(), abs(float(env.z_top[i]) - st.z_top),
                    np.abs(env.bend_cmd[i].cpu().numpy() - st.bend).max(),
                    np.abs(jp[i, env._q_to_joint.cpu().numpy()] - q).max())
        pipe_pos = env.pipe.data.root_pos_w[i] - env.scene.env_origins[i]
        worst = max(worst, float(torch.abs(pipe_pos[:2].double() - env.pipe_xy[i]).max()), abs(float(pipe_pos[2]) - float(env.z_top[i])))
    print(f"[2] resets: {env.num_envs} envs ({sum(m == 'rig' for m in env.env_modes)} rig), "
          f"{int(env.assisted.sum())} assisted starts, max difference to the MuJoCo sampler {worst:.2e}")
    assert worst < 1e-5


def replay(env, ref_path):
    if not os.path.exists(ref_path):
        print(f"[3] replay: no reference file {ref_path} - run tests/compare_with_mujoco.py first")
        return None
    data = np.load(ref_path)
    episodes, a0, o0 = [], 0, 0
    for k, length in enumerate(data["lengths"]):
        episodes.append(dict(mode=str(data["mode"][k]), outcome=str(data["outcome"][k]), pipe_xy=data["pipe_xy"][k],
                             z_top=float(data["z_top"][k]), bend=data["bend"][k], elev=float(data["elev"][k]),
                             actions=data["actions"][a0:a0 + length], obs=data["obs"][o0:o0 + length + 1],
                             rewards=data["rewards"][a0:a0 + length]))
        a0, o0 = a0 + length, o0 + length + 1
    n = min(len(episodes), env.num_envs)
    episodes = episodes[:n]
    for k, ep in enumerate(episodes):
        assert ep["mode"] == env.env_modes[k], "episode / env pipe modes differ"
    env.cfg.max_episode_steps = 150
    env.reset()
    ids = list(range(n))
    env.set_start_state(ids, [e["pipe_xy"] for e in episodes], [e["z_top"] for e in episodes],
                        [e["bend"] for e in episodes], [e["elev"] for e in episodes])
    obs = env._get_observations()["policy"].cpu().double().numpy()
    d_obs0 = max(np.abs(obs[k] - episodes[k]["obs"][0]).max() for k in ids)

    lengths = [len(e["actions"]) for e in episodes]
    active = np.ones(n, dtype=bool)
    tip_err, obs_err, rew_err, cmd_err = [], [], [], []
    contact_confusion = np.zeros((2, 2), dtype=int)       # [mujoco contact][isaac contact]
    count_table = np.zeros((5, 5), dtype=int)             # min(#contacts, 4): [mujoco][isaac]
    outcome_pairs = []
    isaac_len = np.zeros(n, dtype=int)
    for t in range(max(lengths)):
        act = np.zeros((env.num_envs, 7), dtype=np.float32)
        for k in ids:
            if active[k] and t < lengths[k]:
                act[k] = episodes[k]["actions"][t]
        obs_d, rew, term, trunc, extras = env.step(torch.as_tensor(act, device=env.device))
        final_obs = extras["terminal_obs"].cpu().double().numpy()
        rew = rew.cpu().double().numpy()
        done = (term | trunc).cpu().numpy()
        outcome = extras["done_info"]["outcome"].cpu().numpy()
        for k in ids:
            if not active[k]:
                continue
            mj_obs = episodes[k]["obs"][t + 1]
            o = final_obs[k]
            tip_err.append(np.linalg.norm(o[0:3] - mj_obs[0:3]) * 100.0)          # obs is /0.1 m -> mm
            cmd_err.append(np.abs(o[15:22] - mj_obs[15:22]).max())
            obs_err.append(np.abs(o - mj_obs))
            rew_err.append(abs(rew[k] - episodes[k]["rewards"][t]))
            contact_confusion[int(mj_obs[25] > 0), int(o[25] > 0)] += 1
            count_table[int(round(mj_obs[25] * 4)), int(round(o[25] * 4))] += 1
            mj_done = t + 1 == lengths[k]
            if done[k] or mj_done:
                names = ["success", "unstable", "rim_hit", "missed_pipe", "timeout"]
                isaac_out = names[outcome[k]] if done[k] else "running"
                outcome_pairs.append((episodes[k]["outcome"], isaac_out, lengths[k], t + 1))
                isaac_len[k] = t + 1
                active[k] = False
        if not active.any():
            break
    tip_err, rew_err = np.array(tip_err), np.array(rew_err)
    same = sum(a == b for a, b, _, _ in outcome_pairs)
    print(f"[3] replay of {n} MuJoCo episodes ({sum(lengths)} steps):")
    print(f"    start obs max |diff| {d_obs0:.2e}; commands (bend/elev obs) max |diff| {max(cmd_err):.2e}")
    print(f"    tip position diff: mean {tip_err.mean():.3f} mm, 95% {np.percentile(tip_err, 95):.3f} mm, max {tip_err.max():.3f} mm")
    print(f"    reward diff: mean {rew_err.mean():.4f}, 95% {np.percentile(rew_err, 95):.4f} (incl. the +100 / -30 terminal terms)")
    print(f"    wall-contact steps (rows MuJoCo no/yes, cols Isaac no/yes): {contact_confusion.tolist()}")
    print(f"    contact count min(n,4) (rows MuJoCo 0..4, cols Isaac 0..4): {count_table.tolist()}")
    obs_err = np.array(obs_err)
    groups = {"tip": (0, 3), "tip dir": (3, 6), "Seg13": (6, 9), "Seg11": (9, 12), "Seg6": (12, 15), "bend": (15, 21),
              "elev": (21, 22), "pipe offset": (22, 24), "height": (24, 25), "contact": (25, 26), "prev action": (26, 33)}
    print("    obs diff per term, 95th percentile: " + ", ".join(
        f"{k} {np.percentile(obs_err[:, a:b].max(axis=1), 95):.1e}" for k, (a, b) in groups.items()))
    print(f"    episode outcome identical in {same}/{len(outcome_pairs)} episodes")
    for mj, isa, lm, li in outcome_pairs:
        if mj != isa:
            print(f"      MuJoCo {mj:12s} after {lm:3d} steps | Isaac {isa:12s} after {li:3d} steps")
    return tip_err


def random_actions(env, steps):
    env.cfg.max_episode_steps = 400
    env.assist_prob = 0.4
    env.reset()
    rng = np.random.default_rng(0)
    outcomes = {}
    t0 = time.time()
    nonfinite = 0
    rew_min, rew_max = np.inf, -np.inf
    for _ in range(steps):
        act = torch.as_tensor(rng.uniform(-1, 1, size=(env.num_envs, 7)), dtype=torch.float32, device=env.device)
        obs, rew, term, trunc, extras = env.step(act)
        nonfinite += int((~torch.isfinite(obs["policy"])).any(dim=1).sum())
        r = rew.cpu().numpy()
        rew_min, rew_max = min(rew_min, r.min()), max(rew_max, r.max())
        for o in extras["done_info"]["outcome"].cpu().numpy():
            if o >= 0:
                name = ["success", "unstable", "rim_hit", "missed_pipe", "timeout"][o]
                outcomes[name] = outcomes.get(name, 0) + 1
    dt = time.time() - t0
    print(f"[4] random actions: {steps} steps x {env.num_envs} envs in {dt:.1f} s "
          f"({steps * env.num_envs / dt:.0f} env-steps/s), non-finite obs rows {nonfinite}, "
          f"reward range [{rew_min:.2f}, {rew_max:.2f}], episode ends {outcomes}")
    assert nonfinite == 0


def check_snapshot(env):
    env.cfg.max_episode_steps = 400
    env.reset()
    rng = np.random.default_rng(3)
    acts = [torch.as_tensor(rng.uniform(-1, 1, size=(env.num_envs, 7)), dtype=torch.float32, device=env.device)
            for _ in range(40)]
    for a in acts[:10]:
        env.step(a)
    snap = env.snapshot()
    runs = []
    for _ in range(3):
        obs = [env.step(a)[0]["policy"].clone() for a in acts[10:]]
        for _ in range(30):                      # something else in between (evaluation episodes)
            env.step(torch.zeros_like(acts[0]))
        env.restore(snap)
        runs.append(torch.stack(obs))
    d_live = (runs[0] - runs[1]).abs().amax(dim=(0, 2))       # live state vs restored state
    d_rest = (runs[1] - runs[2]).abs().amax(dim=(0, 2))       # restored vs restored
    print(f"[5] snapshot/restore over 30 steps: restored vs restored max obs diff {float(d_rest.max()):.2e}; "
          f"live vs restored: median env {float(d_live.median()):.2e}, max {float(d_live.max()):.2e}, "
          f"envs > 1e-3: {int((d_live > 1e-3).sum())}/{env.num_envs}")


def main():
    cfg = VerticalPipeEnvCfg()
    cfg.scene.num_envs = N_ENVS
    cfg.pipe_modes = ("rig", "random")
    if args.touch_tol is not None:
        cfg.contact_touch_tolerance = args.touch_tol
    if args.device is not None:
        cfg.sim.device = args.device
    env = VerticalPipeEnv(cfg)
    print(f"env: {env.num_envs} envs, device {env.device}, sim dt {env.physics_dt}, decimation {cfg.decimation}, "
          f"step dt {env.step_dt}")
    print(f"     joints {env.robot.num_joints} ({env.robot.joint_names[:3]}...), bodies {env.robot.num_bodies}, "
          f"contact sensors {env._contact_view.sensor_count}, touch tol {cfg.contact_touch_tolerance * 1000:.2f} mm")
    if not args.only_replay:
        check_kinematics(env)
        check_resets(env)
    for tol in (args.rim_tol or [cfg.rim_touch_tolerance]):
        env.cfg.rim_touch_tolerance = tol
        print(f"--- rim touch tolerance {tol * 1000:.2f} mm")
        replay(env, args.reference)
    if not args.only_replay:
        random_actions(env, args.random_steps)
        check_snapshot(env)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
