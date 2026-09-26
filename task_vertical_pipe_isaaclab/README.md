# Vertical-pipe task in Isaac Lab (port of the MuJoCo version)

Port of `D:\mujoco\Continuum_MuJoCo\task_vertical_pipe` (continuum robot bends into an S-shape
and feeds down through a vertical pipe) to Isaac Sim 5.1 / Isaac Lab 2.3.2, trained with rsl_rl.
Observation, action, reward, termination, reset / randomization, `dt` and `frame_skip` are the
ones of the MuJoCo env; the PPO update and hyperparameters are the ones of its SB3 trainer.

| file | content |
|---|---|
| `constants.py` | task constants, copied 1:1 from `vertical_pipe_env.py` / `continuum_algorithm.py` |
| `kinematics.py` | MJCF kinematic tree + forward kinematics identical to `mj_kinematics` (numpy) |
| `reset_logic.py` | start-state sampling (`reset`, `_place_pipe`, `_assisted_start`), same numpy draws |
| `mdp.py` | action -> commands, measurements, observation (33), reward, termination (torch) |
| `vertical_pipe_env.py` | Isaac Lab `DirectRLEnv` + scene / sim configuration |
| `convert_mjcf_to_usd.py` | MJCF -> USD (robot) and the pipe USDs |
| `ppo_sb3.py`, `agent_cfg.py`, `pipe_runner.py` | rsl_rl PPO with the SB3 update rule, config, runner with eval / checkpoints |
| `train_vertical_pipe.py`, `play_vertical_pipe.py` | training / viewer, statistics, video |
| `import_sb3_policy.py` | SB3 model of the MuJoCo trainer (`.zip`) -> rsl_rl checkpoint (same networks) |
| `tests/compare_with_mujoco.py` | checks against the MuJoCo env (runs in the MuJoCo venv) |
| `tests/check_isaac_env.py` | checks of the Isaac env, replay of MuJoCo trajectories in PhysX |
| `assets/mjcf/ContinuumRobot_Native.xml` | source MJCF (copy of `urdf/ContinuumRobot_Native.xml`) |

## Run

```cmd
cd D:\Continuum-Robot-Riclab-\task_vertical_pipe_isaaclab
set ISAAC=D:\Isaacsim\env_isaaclab\Scripts\python.exe

:: 0. (optional) engine-independent parts vs the MuJoCo env, writes tests/data/mujoco_reference.npz
D:\mujoco\Continuum_MuJoCo\.venv\Scripts\python.exe tests\compare_with_mujoco.py

:: 1. MJCF -> USD (assets/generated/, not in git). --mesh-dir: folder with the .obj meshes
%ISAAC% convert_mjcf_to_usd.py --mesh-dir D:\mujoco\Continuum_MuJoCo\urdf

:: 2. checks in Isaac Sim (kinematics, resets, replay of the MuJoCo episodes, random actions)
%ISAAC% tests\check_isaac_env.py

:: 3. train (3M steps, 16 envs, rig + random scenes; same options as the MuJoCo trainer)
%ISAAC% train_vertical_pipe.py
%ISAAC% train_vertical_pipe.py --pipe-mode rig --timesteps 1000000
%ISAAC% train_vertical_pipe.py --resume models\ppo_vpipe\best_model.pt --contact-penalty 0.5 --run-name ppo_vpipe_ft
%ISAAC% train_vertical_pipe.py --n-envs 12 --stop-at 200000 :: short run, lr schedule still over 3M steps
tensorboard --logdir logs

:: 4. watch / evaluate
%ISAAC% play_vertical_pipe.py                               :: Isaac Sim window, rig then random pipe
%ISAAC% play_vertical_pipe.py --headless --episodes 100     :: success statistics
%ISAAC% play_vertical_pipe.py --video --episodes 5          :: MP4 with HUD in videos\

:: the policy trained in MuJoCo, in Isaac Sim (or as a start for --resume)
%ISAAC% import_sb3_policy.py D:\mujoco\Continuum_MuJoCo\task_vertical_pipe\models\ppo_vpipe_wide\final_model.zip models\mujoco_ppo_vpipe_wide\model.pt
%ISAAC% play_vertical_pipe.py --checkpoint models\mujoco_ppo_vpipe_wide\model.pt
```

All scripts append `--/app/vulkan=false` (DirectX 12, as the other Isaac scripts on this machine)
and import torch / tensordict / rsl_rl before Kit starts (Windows DLL order).

## What is the same as MuJoCo

| | MuJoCo (`task_vertical_pipe`) | Isaac Lab |
|---|---|---|
| timestep / frame_skip | `opt.timestep` 0.002 s, `N_SUBSTEPS` 10 | `sim.dt` 0.002 s, `decimation` 10 (20 ms, 50 Hz) |
| episode | 400 steps, truncation | `max_episode_steps` 400 (`episode_length_s` 8 s), `time_outs` |
| action (7) | bend-vector rates of 3 sections + elevator rate | `mdp.update_commands` (same rates, clipping, THETA_MAX) |
| observation (33) | `_get_obs` | `mdp.observation` (same terms, order and scaling) |
| reward | potential progress, time, wall contact, smoothness, +100 / -30 | `mdp.step_reward` |
| termination | success / unstable / rim_hit / missed_pipe | `mdp.outcome` |
| reset | pipe placement, assisted starts (40 % in training), 50 attempts | `reset_logic.py`; env i uses `default_rng(seed + i)` like SB3's seeding |
| scenes | `--pipe-mode mixed`: even envs rig, odd envs random | `pipe_modes=("rig", "random")`, same split |
| parallel envs | 16 (`SubprocVecEnv`, code default; the model in the task README was trained with 12) | 16 (`--n-envs`) |
| physics | gravity 0, servos kp 100 / kv 2 (elevator 3e4 / 1.2e3), armature 0.01, damping 0.5, force limit 20 N m, friction 0.3 | same values on PhysX implicit drives / materials |

Checked (`tests/`):
- the MJCF given to the importer compiles in MuJoCo to the model the MuJoCo env simulates (masses,
  joints, actuators, colliders; 300 steps with random controls differ by 3e-17);
- `kinematics.py` vs `mj_kinematics`: 1e-15 m; PhysX link poses of the USD vs `kinematics.py`: 0.005 mm;
- 3600 resets (1661 assisted) sampled like MuJoCo for the same seeds: identical (1e-15);
- `mdp.py` fed with MuJoCo states over 6075 steps (all outcomes, 937 wall-contact steps):
  observations bit-identical, rewards within 7e-15, no episode-end mismatch;
- replay of 72 MuJoCo episodes (same start state and actions) in PhysX: tip position differs by
  0.48 mm on average (95 %: 0.80 mm), reward by < 0.014 on 95 % of the steps, wall-contact steps
  agree on 900 of 905, the episode outcome is the same in 67/72 episodes (the others: 3 failures of
  another kind a few steps earlier - the tip passes within 1 mm of the rim - and 2 long random-action
  episodes that diverge after ~100 steps of wall contact). All observation terms agree (95 %: body
  positions < 0.7 mm, commands / pipe offset / previous action exact) except the contact count
  (`min(n, 4) / 4`), where the two engines produce different numbers of points per touch (mean
  difference 0.8 contacts);
- the policy trained in MuJoCo (`ppo_vpipe_wide/final_model.zip`, via `import_sb3_policy.py`) run in
  Isaac Sim (`play_vertical_pipe.py --headless --episodes 100 --seed 9000`):

  | scene | Isaac Sim | MuJoCo (task README) |
  |---|---|---|
  | rig | 99 % success (1 rim hit), 2.39 s, clearance median 4.0 mm, 100 % without wall contact | 100 %, 2.4 s, 4.0 mm, 100 % |
  | random 0-17 cm | 99 % (1 rim hit), 1.86 s, clearance median 5.9 mm, 81 % without wall contact | 99 % (1 rim hit), 83 % |

## MJCF -> USD

`convert_mjcf_to_usd.py` rebuilds the model the MuJoCo env simulates (the env edits the MJCF in code:
it removes the mesh colliders, adds one collision cylinder per segment, 31 position servos, armature
and damping), converts it with the Isaac Sim MJCF importer (`isaaclab.sim.converters.MjcfConverter`)
into `continuum_physics.usd` (colliders only, for training) and `continuum_visual.usd` (+ CAD meshes),
and writes `pipe_rig.usd` / `pipe_random.usd` (24 box staves on a kinematic body, bore 100 / 70 mm).

What the importer does not carry over, and the replacement:

| MuJoCo | Isaac Sim / Isaac Lab |
|---|---|
| tendons, equality constraints, `cable` / `flexcomp` plugins | none in this model (the continuum body is already a chain of 15 rigid universal joints, joint angle = section bend / 5). For other models: fixed tendons -> PhysX fixed tendons (the importer converts them), `equality joint` -> PhysX tendon / mimic joint, `weld` / `connect` -> fixed / spherical joint, cable / flexcomp -> chain of rigid capsules with drives, or a PhysX deformable body |
| `<option>` (timestep, gravity 0, implicitfast, Newton, pyramidal cone) | `SimulationCfg(dt=0.002, gravity=0)`, PhysX TGS, 8 position / 1 velocity iterations; drives are implicit like implicitfast |
| position servo `kv` (dropped by the importer) and joint damping | `ImplicitActuatorCfg`: stiffness kp, damping kv + 0.5, effort limit = `actuatorfrcrange` (20 N m), armature 0.01 |
| contact parameters: `solref`, `solimp`, `margin`, `condim`, torsional / rolling friction | PhysX rigid contacts: friction 0.3 static = dynamic (only sliding friction is used with condim 3), restitution 0, contact offset 2 mm, rest offset 0. MuJoCo's soft contacts let the tip sink 0.5-1.3 mm into the rim before `rim_hit`; PhysX does not penetrate, so a point outside the rim-radius counts as a hit up to 1 mm of separation (`rim_touch_tolerance`; 1 mm gives the best outcome agreement: 0.1 mm 59/72, 0.5 mm 64/72, 1 mm 67/72, 2 mm 65/72) and an inner-wall point counts as contact up to 0.1 mm (`contact_touch_tolerance`) |
| `contype` / `conaffinity` (segments never collide with each other) | importer: filtered pairs; `enabled_self_collisions=False` |
| mocap pipe body | kinematic rigid body, moved at reset (`write_root_pose_to_sim`) |
| world-body frame geoms + slide joint to the world | jointless `base_link` root (fixed joint), the elevator is a prismatic articulation joint; the importer anchors the fixed joint at the world origin, the env moves each env's anchor to its env origin |
| extra `worldBody` articulation root, visual copies of the collision cylinders, instanced colliders | removed / hidden / de-instanced by the converter |
| contact list `data.contact` | PhysX contact points of every segment with the pipe of its env (tensor contact view). MuJoCo gives 1-2 contacts per segment / stave pair, PhysX a patch of up to ~30 points: the wall-contact count uses one contact per (segment, stave) pair, the stave found from the point's angle around the pipe axis |
| `mj_forward` + `ncon == 0` test of a start pose | `kinematics.pipe_overlap` (sampled cylinder / box overlap); gives the same accept / reject as MuJoCo on all 1800 test resets |
| `mjWARN_BADQACC` ("unstable") | non-finite joint state or joint speed > 1000 rad/s (m/s) |

Small remaining differences: PhysX vs MuJoCo integration and contact dynamics (see the replay numbers);
MuJoCo's body poses after `mj_step` belong to the start of the last substep, PhysX's to its end (2 ms);
the servo force limit also clips the passive damping part in PhysX.

## Training (rsl_rl)

`ppo_sb3.SB3PPO` is rsl_rl's PPO with the update of Stable-Baselines3 PPO, the trainer of the MuJoCo
version; rsl_rl provides the rollout storage, runner and logging.

| SB3 (MuJoCo trainer) | here |
|---|---|
| `learning_rate=linear_schedule(3e-4, 1e-5)` | same schedule, per timestep (resume: 1e-4 -> 1e-5) |
| `n_steps=512`, `batch_size=1024`, `n_epochs=10` | `num_steps_per_env=512`, `num_mini_batches=8` (16 envs), 10 epochs |
| `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0`, `vf_coef=0.5`, `max_grad_norm=0.5` | same |
| `net_arch pi=[256,256], vf=[256,256]`, Tanh, `log_std_init=-0.5`, ortho init | same (`init_std=exp(-0.5)`, log-std parameter, SB3 orthogonal init) |
| Adam eps 1e-5, minibatches reshuffled every epoch, advantage normalised per minibatch, unclipped value loss, one grad-norm clip | same (rsl_rl defaults differ; overridden in `SB3PPO`) |
| truncated episodes bootstrapped with V(terminal observation) | same (`extras["terminal_obs"]`; rsl_rl would use V(s_t)) |
| `CheckpointCallback` 200k, `EvalCallback` 50k / 20 deterministic episodes / `best_model`, `PipeStatsCallback` | `PipeRunner` (same frequencies and TensorBoard tags `eval/*`, `pipe/*`) |
| 3M steps, 16 envs, seed 0, PPO on CPU | same (`--ppo-device cpu`; PhysX on `--device cuda:0`) |

Isaac Sim runs one simulation per process, so there is no separate eval env: the evaluation saves the
state of all training envs, runs the episodes (no assisted starts, own random streams) and restores
the state afterwards.

### Learning speed depends strongly on the seed (also in MuJoCo)

The curve in the MuJoCo README (90 % eval success at 200k steps) comes from one run with 12 envs.
The original MuJoCo trainer, rerun unchanged, gives at 200k steps (training success of the last 200
episodes):

| MuJoCo SB3 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 12 envs | 76 % | 0 % (hovers above the pipe) | 18 % |
| 16 envs (code default) | 16 % | 0 % (hovers) | 21 % |

Isaac Lab, same settings: 16 envs / seed 0 with rsl_rl: 0 % at 150k; SB3 itself on the Isaac env
(A/B check of the rsl_rl port): 0 % at 200k, with the same early statistics as MuJoCo 16 envs / seed 0
(50k steps: 0 % success, 7 % rim hits, 92 % time-outs in both). 12 envs / seed 0 with rsl_rl: see
"Results" below. Several seeds (or the MuJoCo policy as a start, `--resume`) are needed to compare
learning speed; one run is not enough in either simulator.

### Results of the short training check

`train_vertical_pipe.py --n-envs 12 --stop-at 200000` (seed 0, learning-rate schedule over 3M steps,
25.9 min on the GPU, 20 deterministic evaluation episodes every 50k steps):

| steps | 55k | 104k | 154k | 203k |
|---|---|---|---|---|
| Isaac Lab / rsl_rl: eval success | 0 % | 0 % | 15 % | 80 % |
| Isaac Lab / rsl_rl: eval mean reward | -8.8 | -5.8 | 14.4 | 76.2 |
| MuJoCo README run (12 envs): eval success | 0 % (50k) | 15 % (100k) | 75 % (150k) | 90 % (200k) |

`best_model.pt` of that run, 100 episodes per scene: rig 92 % success (8 time-outs), random 64 %
(13 rim hits, 23 time-outs). It still slides along the wall (68 wall-contact steps per rig episode)
and is slow (5.7 s); it has seen 7 % of the 3M training steps.

## Notes

- PhysX GPU pipeline only: on the CPU pipeline of Isaac Sim 5.1 the kinematic pipe is moved back to
  its USD pose every step and the tensor contact view returns no contacts (`VerticalPipeEnv` refuses
  `sim.device="cpu"`).
- Speed on this laptop (RTX 5070 Laptop, Windows): ~220 env-steps/s with 16 envs (3M steps ~ 4 h),
  ~920 env-steps/s with 72 envs; the MuJoCo trainer runs ~2300 steps/s. With 16 small articulations
  the GPU pipeline is dominated by its fixed cost per physics step; more envs use the GPU far better
  but change the training setup (batch size per update).
- The random scene's pipe has no legs in the viewer (MuJoCo draws three decorative legs).
- `env.auto_reset = False` keeps finished envs in their final pose until `env.reset_envs(ids)`
  (used by the viewer / video).
- Outputs (`models/`, `logs/`, `videos/`, `assets/generated/`, `tests/data/`) are not in git.
