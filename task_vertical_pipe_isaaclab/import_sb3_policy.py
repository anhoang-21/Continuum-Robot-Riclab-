"""
==============================================================================
import_sb3_policy.py - SB3 PPO model of the MuJoCo trainer -> rsl_rl checkpoint
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe import_sb3_policy.py ^
        D:\\mujoco\\Continuum_MuJoCo\\task_vertical_pipe\\models\\ppo_vpipe_wide\\final_model.zip ^
        models\\mujoco_ppo_vpipe_wide\\model.pt

Both use the same networks (pi / vf: 33 -> 256 -> 256 -> 7 / 1, Tanh, state-independent log std),
so the weights are copied one to one. The result can be watched with
    play_vertical_pipe.py --checkpoint models\\mujoco_ppo_vpipe_wide\\model.pt
or fine-tuned with train_vertical_pipe.py --resume (the Adam state is not transferred).
Only policy.pth is read from the zip (no unpickling, so numpy versions do not matter).
==============================================================================
"""

import argparse
import io
import os
import sys
import zipfile

import torch
from tensordict import TensorDict

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from rsl_rl.models import MLPModel  # noqa: E402

from agent_cfg import make_agent_cfg  # noqa: E402

SB3_TO_RSL = {
    "actor": {"mlp_extractor.policy_net.0": "mlp.0", "mlp_extractor.policy_net.2": "mlp.2", "action_net": "mlp.4"},
    "critic": {"mlp_extractor.value_net.0": "mlp.0", "mlp_extractor.value_net.2": "mlp.2", "value_net": "mlp.4"},
}


def build(name):
    cfg = make_agent_cfg(1, 1)[name]
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items() if k != "class_name"}
    obs = TensorDict({"policy": torch.zeros(1, 33)}, batch_size=[1])
    return MLPModel(obs, {"actor": ["policy"], "critic": ["policy"]}, name, 7 if name == "actor" else 1, **cfg)


def main():
    parser = argparse.ArgumentParser(description="Convert an SB3 PPO .zip (MuJoCo trainer) into an rsl_rl checkpoint")
    parser.add_argument("sb3_zip")
    parser.add_argument("output")
    args = parser.parse_args()

    with zipfile.ZipFile(args.sb3_zip) as z:
        sb3 = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    out = {}
    for name, mapping in SB3_TO_RSL.items():
        model = build(name)
        state = model.state_dict()
        for src, dst in mapping.items():
            for p in ("weight", "bias"):
                if state[f"{dst}.{p}"].shape != sb3[f"{src}.{p}"].shape:
                    raise ValueError(f"shape mismatch {src}.{p} -> {dst}.{p}")
                state[f"{dst}.{p}"] = sb3[f"{src}.{p}"].clone()
        if name == "actor":
            state["distribution.log_std_param"] = sb3["log_std"].clone()
        model.load_state_dict(state, strict=True)
        out[f"{name}_state_dict"] = model.state_dict()
    out.update(iter=0, infos={"source": os.path.abspath(args.sb3_zip)})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print(f"{args.sb3_zip} -> {args.output} (actor + critic, action std {sb3['log_std'].exp().numpy().round(3)})")


if __name__ == "__main__":
    main()
