"""
==============================================================================
compare_with_mujoco.py - Check the engine-independent parts of the port against
the original MuJoCo env (run with the MuJoCo venv, no Isaac Sim needed)
==============================================================================
    D:\\mujoco\\Continuum_MuJoCo\\.venv\\Scripts\\python.exe tests/compare_with_mujoco.py

1. kinematics.py vs mj_kinematics on random joint configurations
2. reset_logic.py vs VerticalPipeEnv.reset for the same seeds (pipe placement,
   start bend / elevator, including assisted starts and consecutive resets)
3. mdp.py vs VerticalPipeEnv.step: MuJoCo is stepped with random actions and the
   torch functions are fed the MuJoCo state (body poses, contact points); the
   observations, rewards and episode ends must match.
4. writes tests/data/mujoco_reference.npz: start states + actions + MuJoCo
   observations / rewards, replayed in Isaac Sim by tests/check_isaac_env.py.
==============================================================================
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, TASK_DIR)

import constants as C  # noqa: E402
import mdp  # noqa: E402
from kinematics import ContinuumKinematics  # noqa: E402
from reset_logic import ResetSampler  # noqa: E402


def load_mujoco_env(mujoco_dir):
    sys.path.insert(0, os.path.join(mujoco_dir, "task_vertical_pipe"))
    import mujoco  # noqa: F401
    from vertical_pipe_env import VerticalPipeEnv
    return VerticalPipeEnv


def check_kinematics(env, kin, n=300, seed=0):
    import mujoco
    rng = np.random.default_rng(seed)
    m, d = env.model, env.data
    names = [n_ for n_ in kin.body_names]
    mj_ids = [m.body(n_).id for n_ in names]
    mj_q = [m.joint(n_).qposadr[0] for n_ in kin.joint_names]
    worst_p, worst_r = 0.0, 0.0
    for _ in range(n):
        q = rng.uniform(kin.joint_range[:, 0], kin.joint_range[:, 1])
        d.qpos[:] = 0.0
        d.qpos[mj_q] = q
        mujoco.mj_kinematics(m, d)
        xpos, xmat = kin.forward(q)
        worst_p = max(worst_p, np.abs(xpos - d.xpos[mj_ids]).max())
        worst_r = max(worst_r, np.abs(xmat - d.xmat[mj_ids].reshape(-1, 3, 3)).max())
    print(f"[1] kinematics: {n} random configs, max |dpos| = {worst_p:.2e} m, max |dR| = {worst_r:.2e}")
    print(f"    axis_xy {kin.axis_xy} vs {env.axis_xy}, tip_z0 {kin.tip_z0:.9f} vs {env.tip_z0:.9f}")
    assert worst_p < 1e-9 and worst_r < 1e-9
    assert np.allclose(kin.axis_xy, env.axis_xy, atol=1e-12) and abs(kin.tip_z0 - env.tip_z0) < 1e-12


def check_generated_mjcf(VerticalPipeEnv, mesh_dir):
    """The MJCF handed to the Isaac Sim importer must compile to the model the MuJoCo env simulates."""
    import mujoco
    from convert_mjcf_to_usd import GEN_MJCF_DIR, build_import_mjcf

    env = VerticalPipeEnv(pipe_mode="rig")            # training model (no meshes)
    me, de = env.model, env.data
    for name, visual in (("continuum_physics", False), ("continuum_visual", True)):
        path = build_import_mjcf(C.SOURCE_MJCF, os.path.join(GEN_MJCF_DIR, f"{name}.xml"), visual, mesh_dir)
        mg = mujoco.MjModel.from_xml_path(path)
        dg = mujoco.MjData(mg)
        errs = {}
        assert mg.nq == me.nq and mg.nu == me.nu, (mg.nq, me.nq, mg.nu, me.nu)
        # option
        errs["option"] = max(abs(mg.opt.timestep - me.opt.timestep), np.abs(mg.opt.gravity - me.opt.gravity).max(),
                             float(mg.opt.integrator != me.opt.integrator))
        # bodies: mass / inertia / inertial frame
        e = 0.0
        for b in range(1, me.nbody):
            nm = me.body(b).name
            if nm == "pipe":
                continue
            g = mg.body(nm)
            e = max(e, abs(g.mass[0] - me.body_mass[b]), np.abs(g.inertia - me.body_inertia[b]).max(),
                    np.abs(g.ipos - me.body_ipos[b]).max(), np.abs(g.iquat - me.body_iquat[b]).max())
        errs["bodies"] = e
        # joints, dofs, actuators
        e = 0.0
        for j in range(me.njnt):
            nm = me.joint(j).name
            jg = mg.joint(nm)
            e = max(e, float(jg.type[0] != me.jnt_type[j]), np.abs(jg.axis - me.jnt_axis[j]).max(),
                    np.abs(jg.range - me.jnt_range[j]).max(), float(jg.limited[0] != me.jnt_limited[j]),
                    np.abs(mg.jnt_actfrcrange[jg.id] - me.jnt_actfrcrange[j]).max(),
                    float(mg.jnt_actfrclimited[jg.id] != me.jnt_actfrclimited[j]),
                    abs(mg.dof_armature[jg.dofadr[0]] - me.dof_armature[me.jnt_dofadr[j]]),
                    abs(mg.dof_damping[jg.dofadr[0]] - me.dof_damping[me.jnt_dofadr[j]]))
        for a in range(me.nu):
            nm = me.actuator(a).name
            ag = mg.actuator(nm)
            e = max(e, np.abs(ag.gainprm[:3] - me.actuator_gainprm[a, :3]).max(),
                    np.abs(ag.biasprm[:3] - me.actuator_biasprm[a, :3]).max(),
                    float(mg.joint(ag.trnid[0]).name != me.joint(me.actuator_trnid[a, 0]).name),
                    float(ag.ctrllimited[0] != me.actuator_ctrllimited[a]),
                    float(ag.forcelimited[0] != me.actuator_forcelimited[a]))
        errs["joints/actuators"] = e
        # colliders
        e = 0.0
        for i in range(1, 16):
            ge, gg = me.geom(f"Seg{i}_col"), mg.geom(f"Seg{i}_col")
            e = max(e, np.abs(gg.size - ge.size).max(), np.abs(gg.pos - ge.pos).max(),
                    np.abs(gg.friction - ge.friction).max(), float(gg.contype[0] != ge.contype[0]),
                    float(gg.conaffinity[0] != ge.conaffinity[0]), float(gg.condim[0] != ge.condim[0]),
                    np.abs(gg.solref - ge.solref).max(), np.abs(gg.solimp - ge.solimp).max())
        errs["colliders"] = e
        # same trajectory for the same controls (no pipe contact): qpos after 300 steps
        rng = np.random.default_rng(0)
        mujoco.mj_resetData(me, de)
        mujoco.mj_resetData(mg, dg)
        qe = [me.joint(n).qposadr[0] for n in (mg.joint(k).name for k in range(mg.njnt))]
        ue = [me.actuator(mg.actuator(k).name).id for k in range(mg.nu)]
        de.mocap_pos[:] = [5.0, 5.0, 5.0]                 # move the pipe out of the way
        for _ in range(300):
            ctrl = rng.uniform(-0.2, 0.2, size=mg.nu)
            ctrl[mg.actuator("Elevator").id] = rng.uniform(-0.05, 0.05)
            dg.ctrl[:] = ctrl
            de.ctrl[ue] = ctrl
            mujoco.mj_step(me, de)
            mujoco.mj_step(mg, dg)
        errs["trajectory"] = np.abs(dg.qpos - de.qpos[qe]).max()
        print(f"[0] generated {name}.xml vs env model: " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
        assert max(errs.values()) < 1e-9, errs
    env.close()


def check_resets(VerticalPipeEnv, kin, n_seeds=40, n_consecutive=15):
    worst = 0.0
    n_total = n_assisted = 0
    mismatches = []
    for mode in ("rig", "random"):
        for assist in (0.0, 0.4, 1.0):
            env = VerticalPipeEnv(pipe_mode=mode, assist_prob=assist)
            sampler = ResetSampler(kin, mode, env.pipe_height, env.random_offset)
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)
                for k in range(n_consecutive):
                    env.reset(seed=seed if k == 0 else None)
                    st = sampler.sample(rng, assist)
                    err = max(np.abs(st.pipe_xy - env.pipe_xy).max(), abs(st.z_top - env.z_top),
                              np.abs(st.bend - env.bend_cmd).max(), abs(st.elev - env.elev_cmd))
                    n_total += 1
                    n_assisted += int(st.assisted)
                    if err > 1e-9:
                        mismatches.append((mode, assist, seed, k, err))
                    worst = max(worst, err)
            env.close()
    print(f"[2] resets: {n_total} resets compared ({n_assisted} assisted), max state difference {worst:.2e}, "
          f"{len(mismatches)} mismatches")
    for mm in mismatches[:10]:
        print("    mismatch (mode, assist, seed, reset#, err):", mm)
    return mismatches


def mujoco_state(env):
    """Everything the torch MDP needs from a MuJoCo state (float64 tensors, batch of 1)."""
    d = env.data
    seg_pos = torch.tensor(d.xpos[env.seg_bids][None])
    tip_rot = torch.tensor(d.xmat[env.tip_bid].reshape(1, 3, 3))
    ncon = d.ncon
    pts = np.zeros((0, 3), dtype=np.float64)
    if ncon:
        geoms = d.contact.geom[:ncon]
        on_pipe = env.pipe_geom_mask[geoms].any(axis=1)
        pts = d.contact.pos[:ncon][on_pipe]
    return seg_pos, tip_rot, torch.tensor(pts)


def mujoco_contact_stats(env):
    """(largest radius from the pipe axis, deepest penetration) of the robot-pipe contacts (nan if none)."""
    d = env.data
    if d.ncon == 0:
        return np.nan, np.nan
    on_pipe = env.pipe_geom_mask[d.contact.geom[:d.ncon]].any(axis=1)
    if not on_pipe.any():
        return np.nan, np.nan
    r = np.linalg.norm(d.contact.pos[:d.ncon][on_pipe][:, :2] - env.pipe_xy, axis=1)
    return float(r.max()), float(d.contact.dist[:d.ncon][on_pipe].min())


def check_mdp(VerticalPipeEnv, n_episodes=72, seed=123, record=None):
    rng = np.random.default_rng(seed)
    worst_obs = worst_rew = 0.0
    n_steps = n_contact_steps = 0
    outcomes = {}
    end_mismatch = 0
    inner_r = C.inner_contact_radius()
    for ep in range(n_episodes):
        mode = ("rig", "random")[ep % 2]
        assist = (0.0, 1.0, 1.0)[ep % 3]          # assisted starts put the tip in / near the pipe -> contacts
        # "random": random walk on all 7 actions; "descend": elevator down with bending noise, which
        # produces wall contacts, rim hits, misses and successes
        style = ("random", "descend")[(ep // 2) % 2]
        env = VerticalPipeEnv(pipe_mode=mode, assist_prob=assist, max_episode_steps=150)
        obs, info = env.reset(seed=1000 + ep)
        start = dict(pipe_xy=env.pipe_xy.copy(), z_top=env.z_top, bend=env.bend_cmd.copy(), elev=env.elev_cmd,
                     mode=mode, qpos=env.data.qpos.copy())
        pipe_xy = torch.tensor(env.pipe_xy[None])
        z_top = torch.tensor([env.z_top], dtype=torch.float64)
        z_success = torch.tensor([env.z_success], dtype=torch.float64)
        axis_xy = torch.tensor(env.axis_xy[None])
        bend = torch.tensor(env.bend_cmd[None].copy())
        elev = torch.tensor([env.elev_cmd], dtype=torch.float64)
        prev_action = torch.zeros(1, 7, dtype=torch.float64)

        # observation at reset
        seg_pos, tip_rot, pts = mujoco_state(env)
        n_inner, outer = mdp.classify_contacts(pts, torch.zeros(len(pts), dtype=torch.long), pipe_xy, inner_r, 1)
        meas = mdp.measure(seg_pos, tip_rot, pipe_xy, z_top, z_success, n_inner, outer, C.PLATE_HOLE_RADIUS)
        prev_pot = mdp.potential(meas)
        o = mdp.observation(meas, pipe_xy, z_top, z_success, bend, elev, axis_xy, prev_action)
        worst_obs = max(worst_obs, np.abs(o.numpy()[0] - obs).max())
        actions, mj_obs, mj_rew, mj_con = [], [obs.copy()], [], []
        # smooth-ish random actions: persistent random walk so the robot actually travels
        a = rng.uniform(-1.0, 1.0, size=7)
        done = False
        while not done:
            a = np.clip(0.8 * a + 0.6 * rng.normal(size=7), -1.2, 1.2)     # also exercises the clipping
            if style == "descend":
                a[:6] = np.clip(0.9 * a[:6] + 0.25 * rng.normal(size=6), -0.6, 0.6)
                a[6] = -1.0
            act = a.astype(np.float32)
            obs, rew, term, trunc, info = env.step(act)
            action = torch.tensor(np.clip(act.astype(np.float64), -1.0, 1.0)[None])
            mdp.update_commands(bend, elev, action)
            seg_pos, tip_rot, pts = mujoco_state(env)
            n_inner, outer = mdp.classify_contacts(pts, torch.zeros(len(pts), dtype=torch.long), pipe_xy, inner_r, 1)
            meas = mdp.measure(seg_pos, tip_rot, pipe_xy, z_top, z_success, n_inner, outer, C.PLATE_HOLE_RADIUS)
            unstable = torch.tensor([info["failure"] == "unstable"])
            r, prev_pot, terminated, success, failure = mdp.step_reward(
                meas, prev_pot, action, prev_action, env.contact_penalty, z_success, C.PLATE_HOLE_RADIUS, unstable)
            prev_action = action
            o = mdp.observation(meas, pipe_xy, z_top, z_success, bend, elev, axis_xy, prev_action)
            truncated = (not bool(terminated[0])) and env.current_step >= env.max_episode_steps
            worst_obs = max(worst_obs, np.abs(o.numpy()[0] - obs).max())
            worst_rew = max(worst_rew, abs(float(r[0]) - rew))
            fail_name = C.FAILURE_NAMES[int(failure[0])]
            if bool(terminated[0]) != term or truncated != trunc or fail_name != info["failure"] \
                    or bool(success[0]) != info["is_success"]:
                end_mismatch += 1
            actions.append(act)
            mj_obs.append(obs.copy())
            mj_rew.append(rew)
            mj_con.append(mujoco_contact_stats(env))
            n_steps += 1
            n_contact_steps += int(n_inner[0] > 0)
            done = term or trunc
        key = info["failure"] or ("success" if info["is_success"] else "timeout")
        outcomes[key] = outcomes.get(key, 0) + 1
        if record is not None:
            record.append(dict(start, actions=np.array(actions), obs=np.array(mj_obs), rewards=np.array(mj_rew),
                               contacts=np.array(mj_con),
                               outcome=key, contact_steps=info["contact_steps"]))
        env.close()
    print(f"[3] mdp: {n_episodes} episodes / {n_steps} steps ({n_contact_steps} with inner-wall contact), "
          f"outcomes {outcomes}")
    print(f"    max |obs diff| = {worst_obs:.2e}, max |reward diff| = {worst_rew:.2e}, "
          f"episode-end mismatches = {end_mismatch}")
    assert worst_obs < 1e-5 and worst_rew < 1e-9 and end_mismatch == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mujoco-dir", default=r"D:\mujoco\Continuum_MuJoCo")
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args()

    VerticalPipeEnv = load_mujoco_env(args.mujoco_dir)
    kin = ContinuumKinematics()
    t0 = time.time()
    check_generated_mjcf(VerticalPipeEnv, os.path.join(args.mujoco_dir, "urdf"))
    env = VerticalPipeEnv()
    check_kinematics(env, kin)
    env.close()
    mismatches = check_resets(VerticalPipeEnv, kin, n_seeds=args.seeds)
    record = []
    check_mdp(VerticalPipeEnv, record=record)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    # plain arrays only (no pickle): the Isaac Sim venv has a different numpy major version
    np.savez_compressed(
        os.path.join(out, "mujoco_reference.npz"),
        mode=np.array([e["mode"] for e in record]), outcome=np.array([e["outcome"] for e in record]),
        pipe_xy=np.array([e["pipe_xy"] for e in record]), z_top=np.array([e["z_top"] for e in record]),
        bend=np.array([e["bend"] for e in record]), elev=np.array([e["elev"] for e in record]),
        contact_steps=np.array([e["contact_steps"] for e in record]),
        lengths=np.array([len(e["actions"]) for e in record]),
        actions=np.concatenate([e["actions"] for e in record]), obs=np.concatenate([e["obs"] for e in record]),
        rewards=np.concatenate([e["rewards"] for e in record]),
        contacts=np.concatenate([e["contacts"] for e in record]))
    print(f"reference trajectories -> {os.path.join(out, 'mujoco_reference.npz')}  ({time.time() - t0:.0f} s)")
    if mismatches:
        raise SystemExit("reset mismatches")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
