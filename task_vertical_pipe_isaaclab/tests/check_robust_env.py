"""
==============================================================================
tests/check_robust_env.py - checks of the cable drive under uncertainty (robust_env.py)
==============================================================================
    D:\\Isaacsim\\env_isaacsim62\\Scripts\\python.exe tests\\check_robust_env.py

0. torch only: UncertainCableModel without friction / backlash == cables.CableModel (tensions, nut loads,
   bit for bit); one cable with capstan friction: T_nut / T_robot stays in [exp(-mu theta), exp(mu theta)],
   and the dead band after a reversal of the nut is (lam - 1 / lam) T_r / k_n
1. PhysX, nominal parameters: RobustCableEnv (uncertain model, CUDA graph) and the nominal drive math give
   the same trajectory for the same start state and actions
2. statics with per-env parameters (k_c, T0 per cable, k_j, payload; no friction): PhysX equilibrium ==
   uncertain_cables.StaticSolver
3. capstan friction: a slow triangle sweep of motor M1 leaves a hysteresis loop that grows with mu theta;
   back at u = 0 the frictionless robot is straight again, the robot with friction is not
4. nut backlash: after a reversal the bend does not move while the screw crosses the play
5. stability: "ood" robots, random actions with the pipes in place, many episodes: no instability
==============================================================================
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import tensordict  # noqa: F401
import rsl_rl  # noqa: F401

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

from app_launch import launcher_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--stability-steps", type=int, default=4000)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
args, _ = parser.parse_known_args()
app = AppLauncher(launcher_args(args, cameras=False)).app

import cables as CB  # noqa: E402
import uncertainty as U  # noqa: E402
from robust_env import RobustCableEnv, RobustCableEnvCfg  # noqa: E402
from uncertain_cables import UncertainCableModel  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def bend_of(q):
    q = q.view(-1, 3, 5, 2).sum(dim=2)
    return torch.stack([q[..., 1], -q[..., 0]], dim=-1)


# ---------------------------------------------------------------------------
def check_torch_model():
    print("\n0. torch model (no PhysX)")
    geom = CB.CableGeometry()
    n = 64
    rng = np.random.default_rng(1)
    nom = CB.CableModel(geom)
    ucm = UncertainCableModel(geom, n)
    q = torch.as_tensor(rng.normal(0.0, 0.05, (n, 30)))
    x = torch.as_tensor(rng.uniform(-1, 1, (n, 6)) * CB.MOTOR_RANGE * 0.3)
    ucm.reset_state(torch.arange(n), q @ ucm.G.T, x)
    worst_T = worst_F = 0.0
    for _ in range(50):
        q = q + torch.as_tensor(rng.normal(0.0, 0.004, (n, 30)))
        qd = torch.as_tensor(rng.normal(0.0, 0.2, (n, 30)))
        x = x + torch.as_tensor(rng.normal(0.0, 2e-4, (n, 6)))
        v = torch.as_tensor(rng.normal(0.0, 0.01, (n, 6)))
        Gq, Gqd = q @ ucm.G.T, qd @ ucm.G.T
        ucm.begin_step(Gq, x)
        F_u = ucm.nut_load(Gq, Gqd, x, v)
        F_n = nom.nut_load(nom.tension_from(Gq, x, Gqd, v))
        T_u = ucm.end_step(Gq, Gqd, x, v)
        T_n = nom.tension_from(Gq, x, Gqd, v)
        worst_T = max(worst_T, (T_u - T_n).abs().max().item())
        worst_F = max(worst_F, (F_u - F_n).abs().max().item())
    check("no friction / backlash == cables.CableModel", worst_T == 0.0 and worst_F == 0.0,
          f"(max |dT| {worst_T:.1e} N, max |d nut load| {worst_F:.1e} N)")

    # one cable (cable 0) with friction: fixed arm, the nut pulls in, then pays out again
    mt = 0.3
    ucm = UncertainCableModel(geom, 1)
    prm = U.nominal_params(1)
    prm["mu_theta"][:] = mt
    ucm.set_params(torch.arange(1), prm)
    Gq = torch.zeros(1, 12, dtype=torch.float64)
    x = torch.zeros(1, 6, dtype=torch.float64)
    ucm.reset_state(torch.arange(1), Gq, x)
    zero = torch.zeros(1, 12, dtype=torch.float64)
    path = np.concatenate([np.linspace(0, -1.5e-3, 151), np.linspace(-1.5e-3, 0.0, 151)])   # M1 nut (m)
    Tr, Tn, ratio_ok = [], [], True
    lam = np.exp(mt)
    for xm in path:
        x[0, 0] = xm
        ucm.begin_step(Gq, x)
        T_r = ucm.end_step(Gq, zero, x, torch.zeros(1, 6, dtype=torch.float64))[0, 0].item()
        T_n = ucm.T_n[0, 0].item()
        Tr.append(T_r)
        Tn.append(T_n)
        if T_r > 1e-9:
            ratio_ok &= (1 / lam - 1e-6) <= T_n / T_r <= lam + 1e-6
    Tr, Tn = np.array(Tr), np.array(Tn)
    check("friction: T_nut / T_robot inside the capstan band", bool(ratio_ok),
          f"(max ratio {np.max(Tn / np.maximum(Tr, 1e-9)):.3f}, exp(mu theta) {lam:.3f})")
    # after the reversal T_r stays put until T_n has dropped from lam T_r to T_r / lam
    i_rev = 150
    held = np.nonzero(np.abs(Tr[i_rev:] - Tr[i_rev]) < 1e-6)[0]
    band = (path[i_rev + held[-1]] - path[i_rev]) if len(held) else 0.0
    k_n = CB.CABLE_STIFFNESS / 0.5
    pred = (lam - 1 / lam) * Tr[i_rev] / k_n
    check("friction: dead band after the reversal", abs(band - pred) < 2.5e-5,
          f"({band * 1e3:.3f} mm, predicted (lam - 1/lam) T_r / k_n = {pred * 1e3:.3f} mm)")


# ---------------------------------------------------------------------------
def make_env(level="nominal", n=None, pipes="random"):
    cfg = RobustCableEnvCfg()
    cfg.scene.num_envs = n or args.num_envs
    cfg.uncertainty = level
    cfg.pipe_modes = (pipes,)
    cfg.sim.device = args.device
    cfg.max_episode_steps = 400
    env = RobustCableEnv(cfg)
    env.auto_reset = False
    env.reset()
    return env


def move_pipes_away(env):
    pose = torch.zeros(env.num_envs, 7, device=env.device)
    pose[:, :3] = env.scene.env_origins + torch.tensor([0.0, 0.0, -3.0], device=env.device)
    pose[:, 6] = 1.0
    env.pipe.write_root_pose_to_sim_index(root_pose=pose)


def settle(env, seconds):
    for _ in range(int(seconds / 0.02)):
        env.step(torch.zeros(env.num_envs, 7, device=env.device))
    return env.joint_q(), env.robot.data.joint_vel.torch[:, env._jchain].abs().max().item()


def straight(env, ids=None):
    """All motors (of envs ids) at 0 and the robot straight at rest (the friction elements relaxed)."""
    ids = list(range(env.num_envs)) if ids is None else list(ids)
    idx = torch.as_tensor(ids, device=env.device)
    env.set_motors(idx, torch.zeros(len(ids), 6, device=env.device, dtype=torch.float64))
    env.motors.reset(idx, torch.zeros(len(ids), 6, device=env.device, dtype=torch.float64))
    q = env.statics.equilibrium(torch.zeros(len(ids), 6, dtype=torch.float64), **env.static_params(ids))
    jp = env.robot.data.joint_pos.torch[idx].clone()
    jp[:, env._jchain] = q.to(env.device).float()
    env.robot.write_joint_state_to_sim_index(position=jp, velocity=torch.zeros_like(jp), env_ids=idx)
    env.ucm.reset_state(idx, q.to(env.device) @ env.ucm.G.T, torch.zeros(len(ids), 6, device=env.device, dtype=torch.float64))


def check_same_as_nominal(env):
    print("\n1. nominal parameters: uncertain model (CUDA graph) vs nominal drive math, same start and actions")
    snap = env.snapshot()
    rng = np.random.default_rng(3)
    acts = torch.as_tensor(rng.uniform(-1, 1, (150, env.num_envs, 7)), dtype=torch.float32, device=env.device)

    def run(nominal):
        env.restore(snap)
        move_pipes_away(env)
        ucm, graph = env.ucm, env._cable_graph
        if nominal:                                   # -> CableVerticalPipeEnv._cable_math (nominal model)
            del env.ucm
            env._cable_graph = None
        qs = []
        for a in acts:
            env.step(a)
            qs.append(env.joint_q().clone())
        env.ucm, env._cable_graph = ucm, graph
        return torch.stack(qs), env.tension.clone()

    q_u, T_u = run(False)
    q_n, T_n = run(True)
    dq = (q_u - q_n).abs().max().item()
    check("same trajectory (150 random steps)", dq < 1e-5, f"(max |dq| {dq:.1e} rad, final max |dT| "
          f"{(T_u - T_n).abs().max().item():.1e} N)")


def check_statics(env):
    print("\n2. statics with per-env parameters (k_c, T0 per cable, k_j, payload), no friction / backlash")
    move_pipes_away(env)
    n = env.num_envs
    rng = np.random.default_rng(7)
    h = n // 2
    half = U.UncertaintySet.make("ood").sample(rng, h)
    prm = {k: np.concatenate([v, v], axis=0) for k, v in half.items()}   # robots i and i + n/2 are the same ...
    prm["mu_theta"][:] = 0.0
    prm["backlash"][:] = 0.0
    prm["latency"][:] = 0
    prm["payload"][:h] = 0.050                              # ... except for a 50 g payload on the first half
    prm["payload"][h:] = 0.0
    prm["pos_noise"][:] = 0.0
    prm["tension_noise"][:] = 0.0
    env.randomized = True
    env.set_params(range(n), prm)
    straight(env)
    u = torch.as_tensor(rng.uniform(-0.5, 0.5, (n, 6)) * CB.MOTOR_RANGE / (CB.MOTOR_SECTION + 1), device=env.device)
    u[h:2 * h] = u[:h]                                       # same motors with / without the payload
    env.set_motors(slice(None), u)
    q_sim, vmax = settle(env, 6.0)
    q_mod = env.statics.equilibrium(env.motor.cpu(), q0=q_sim.cpu(), **env.static_params(range(n))).to(env.device)
    err = (q_sim - q_mod).abs().max(dim=1).values
    b_sim, b_mod = bend_of(q_sim), bend_of(q_mod)
    for i in range(min(n, 8)):
        print(f"  env {i}: k_c {prm['k_c'][i].mean():6.0f} T0 {prm['t0'][i].mean():.2f} k_j {prm['k_j'][i]:.2f} "
              f"payload {prm['payload'][i] * 1e3:3.0f} g | bend sim {np.degrees(torch.linalg.norm(b_sim[i], dim=-1).cpu().numpy())} "
              f"model {np.degrees(torch.linalg.norm(b_mod[i], dim=-1).cpu().numpy())} deg  max |dq| {err[i]:.1e}")
    sag_sim = (b_sim[:h] - b_sim[h:2 * h]).abs().max().item()
    sag_mod = (b_mod[:h] - b_mod[h:2 * h]).abs().max().item()
    db = np.degrees((b_sim - b_mod).abs().max().item())
    check("settled", vmax < 2e-2, f"(max |qdot| {vmax:.1e} rad/s)")
    # the cables set the section bends (G is the same for the 5 joints of a section); how a bend is split over
    # the 5 joints is held only by the backbone (k_j 0.5, c_j 2: time constant 4 s) and is still settling
    check("PhysX statics == StaticSolver", err.max().item() < 2e-2 and db < 0.1,
          f"(section bends within {db:.3f} deg, single joints within {err.max().item():.1e} rad)")
    check("payload effect: PhysX == model", abs(sag_sim - sag_mod) < 0.25 * sag_mod + 1e-3,
          f"(50 g changes a section bend by up to {np.degrees(sag_sim):.2f} deg in PhysX, {np.degrees(sag_mod):.2f} deg in the model)")


def check_hysteresis(env):
    print("\n3. capstan friction: slow triangle sweep of M1 (0 -> -12 mm -> 0 -> +12 mm -> 0)")
    move_pipes_away(env)
    n = env.num_envs
    prm = U.nominal_params(n)
    mts = np.linspace(0.0, 0.45, n)
    prm["mu_theta"][:] = mts[:, None]
    env.randomized = True
    env.set_params(range(n), prm)
    straight(env)
    settle(env, 0.5)
    legs = [np.linspace(0, -0.012, 150), np.linspace(-0.012, 0, 150), np.linspace(0, 0.012, 150), np.linspace(0.012, 0, 150)]
    path = np.concatenate(legs)
    bends = []
    for xm in path:
        u = torch.zeros(n, 6, device=env.device, dtype=torch.float64)
        u[:, 0] = xm
        env.set_motors(slice(None), u)
        env.step(torch.zeros(n, 7, device=env.device))
        b = bend_of(env.joint_q())[:, 0]                      # section-1 bend vector
        bends.append(b[:, 0].cpu().numpy())                   # its component along M1 (0 deg)
    q_end, _ = settle(env, 1.0)
    bends = np.array(bends)                                   # (steps, n)
    # loop area in the (u, bend) plane
    area = np.abs(np.trapezoid(bends, path, axis=0)) if hasattr(np, "trapezoid") else np.abs(np.trapz(bends, path, axis=0))
    resid = np.degrees(torch.linalg.norm(bend_of(q_end)[:, 0], dim=-1).cpu().numpy())
    for i in range(0, n, max(1, n // 8)):
        print(f"  mu theta {mts[i]:.2f}: loop area {area[i] * 1e3:.3f} mm rad, section-1 bend back at u = 0: {resid[i]:.2f} deg")
    # (the frictionless loop is the dynamic lag of a 4 mm/s sweep; friction adds the rate-independent part)
    check("frictionless robot returns straight", resid[0] < 0.05, f"({resid[0]:.3f} deg)")
    check("friction leaves a residual bend that grows with mu theta", bool(np.all(np.diff(resid) > -0.01)) and resid[-1] > 0.1,
          f"({resid[-1]:.2f} deg at mu theta {mts[-1]:.2f})")
    check("loop area grows with friction", bool(np.all(np.diff(area) > -0.01 * area[0])) and area[-1] > 2 * area[0],
          f"(x{area[-1] / area[0]:.1f} from mu theta 0 to {mts[-1]:.2f})")
    return path, bends, mts


def check_backlash(env):
    print("\n4. nut backlash: reversal after winding M1 in by 8 mm")
    move_pipes_away(env)
    n = env.num_envs
    prm = U.nominal_params(n)
    plays = np.linspace(0.0, 0.6e-3, n)
    prm["backlash"][:] = plays[:, None]
    env.set_params(range(n), prm)
    straight(env)
    settle(env, 0.5)
    for xm in np.linspace(0, -0.008, 100):
        u = torch.zeros(n, 6, device=env.device, dtype=torch.float64)
        u[:, 0] = xm
        env.set_motors(slice(None), u)
        env.step(torch.zeros(n, 7, device=env.device))
    q0, _ = settle(env, 1.0)
    b0 = bend_of(q0)[:, 0, 0].cpu().numpy()
    u = torch.zeros(n, 6, device=env.device, dtype=torch.float64)
    u[:, 0] = -0.008 + 0.3e-3                               # back by 0.3 mm
    env.set_motors(slice(None), u)
    q1, _ = settle(env, 1.0)
    db = np.degrees(np.abs(bend_of(q1)[:, 0, 0].cpu().numpy() - b0))
    for i in range(0, n, max(1, n // 8)):
        print(f"  play {plays[i] * 1e3:.2f} mm: bend change after 0.3 mm back {db[i]:.3f} deg")
    check("no play: the bend follows the screw", db[0] > 0.2, f"({db[0]:.3f} deg)")
    check("play > 0.6 x 0.3 mm: the bend stays (screw inside the play)", db[-1] < 0.05, f"({db[-1]:.3f} deg)")


def check_stability(env):
    print(f"\n5. 'ood' robots, random actions with the pipes in place, {args.stability_steps} steps with resets")
    env.uset = U.UncertaintySet.make("ood")
    env.randomized = True
    env.auto_reset = True
    env.cfg.max_episode_steps = 100                         # short episodes: many robots
    obs, _ = env.reset()
    n_ep, outcomes, alarms, worst_T = 0, {}, 0, 0.0
    names = ("success", "unstable", "rim_hit", "missed_pipe", "timeout")
    for _ in range(args.stability_steps):
        obs, _, _, _, ex = env.step(torch.rand(env.num_envs, 7, device=env.device) * 2 - 1)
        oc = ex["done_info"]["outcome"].cpu().numpy()
        for o in oc[oc >= 0]:
            outcomes[names[o]] = outcomes.get(names[o], 0) + 1
            n_ep += 1
        alarms += int(env.motors.alarm.any())
        worst_T = max(worst_T, env.tension.max().item())
        if not torch.isfinite(obs["policy"]).all():
            check("finite observations", False)
            break
    print(f"  {n_ep} episodes: {outcomes} | max tension {worst_T:.1f} N | steps with a motor alarm {alarms}")
    check("no instability", outcomes.get("unstable", 0) == 0)


def main():
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    check_torch_model()
    env = make_env("nominal")
    check_same_as_nominal(env)
    check_statics(env)
    check_hysteresis(env)
    check_backlash(env)
    check_stability(env)
    print("\nsummary:", "all checks passed" if not FAILS else f"FAILED: {FAILS}")


if __name__ == "__main__":
    main()
    app.close()
