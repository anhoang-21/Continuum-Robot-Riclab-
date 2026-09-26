"""
==============================================================================
pipe_runner.py - rsl_rl OnPolicyRunner with the callbacks of the MuJoCo trainer
==============================================================================
Same learning loop as rsl_rl.runners.OnPolicyRunner.learn, plus what the SB3
callbacks of task_vertical_pipe/train_vertical_pipe.py do:

  PipeStatsCallback   -> outcome rates of the last 200 training episodes
                         (pipe/success_rate, rim_hit, missed_pipe, unstable, timeout,
                         contact_steps_per_ep, success_ep_len), printed every 30 s
  CheckpointCallback  -> models/<run>/model_<steps>_steps.pt every 200k steps
  EvalCallback        -> every 50k steps: 20 deterministic episodes without assisted
                         starts, eval/mean_reward, eval/success_rate, best_model.pt,
                         evaluations.npz. Isaac Sim runs one simulation per process, so
                         the evaluation uses the training envs: their state is saved,
                         the episodes are run, and the state is restored afterwards.
==============================================================================
"""

from __future__ import annotations

import os
import time
from collections import deque

import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan

OUTCOMES = ("success", "unstable", "rim_hit", "missed_pipe", "timeout")   # index = outcome code of the env


class PipeRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", models_dir=None,
                 eval_freq=50_000, n_eval_episodes=20, checkpoint_freq=200_000, stats_window=200):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.models_dir = models_dir
        self.eval_freq, self.n_eval_episodes, self.checkpoint_freq = eval_freq, n_eval_episodes, checkpoint_freq
        self.outcomes = deque(maxlen=stats_window)
        self.contact_steps = deque(maxlen=stats_window)
        self.success_len = deque(maxlen=stats_window)
        self.num_timesteps = 0
        self.best_mean_reward = -np.inf
        self.eval_log = {"timesteps": [], "results": [], "ep_lengths": [], "successes": []}
        self._last_print = time.time()

    # ------------------------------------------------------------------
    def _record_episodes(self, extras: dict) -> None:
        info = extras["done_info"]
        outcome = info["outcome"]
        ended = (outcome >= 0).nonzero().view(-1)
        if len(ended) == 0:
            return
        oc = outcome[ended].cpu().tolist()
        cs = info["contact_steps"][ended].cpu().tolist()
        ln = info["episode_length"][ended].cpu().tolist()
        for o, c, length in zip(oc, cs, ln):
            self.outcomes.append(OUTCOMES[o])
            self.contact_steps.append(c)
            if o == 0:
                self.success_len.append(length)

    def _log_pipe_stats(self, it: int) -> None:
        if not self.outcomes:
            return
        n = len(self.outcomes)
        writer = self.logger.writer
        if writer is not None:
            for name in ("success", "rim_hit", "missed_pipe", "unstable", "timeout"):
                writer.add_scalar(f"pipe/{name}_rate", self.outcomes.count(name) / n, self.num_timesteps)
            writer.add_scalar("pipe/contact_steps_per_ep", float(np.mean(self.contact_steps)), self.num_timesteps)
            if self.success_len:
                writer.add_scalar("pipe/success_ep_len", float(np.mean(self.success_len)), self.num_timesteps)
        if time.time() - self._last_print > 30:
            self._last_print = time.time()
            rates = " | ".join(f"{k}: {100 * self.outcomes.count(k) / n:5.1f}%"
                               for k in ("success", "rim_hit", "missed_pipe", "unstable", "timeout"))
            print(f"[{self.num_timesteps:>9,} steps] last {n} eps -> {rates}", flush=True)

    # ------------------------------------------------------------------
    def evaluate(self, n_episodes: int, deterministic: bool = True):
        """SB3 evaluate_policy on the training envs (episode targets split over envs like SB3)."""
        env = self.env.unwrapped
        self.alg.eval_mode()
        policy = self.alg.get_policy()
        n = self.env.num_envs
        targets = np.array([(n_episodes + i) // n for i in range(n)])
        counts = np.zeros(n, dtype=int)
        cur_rew = torch.zeros(n)
        cur_len = torch.zeros(n)
        rewards, lengths, successes = [], [], []
        # env buffers are created under inference mode during the rollouts, so everything stays in it
        with torch.inference_mode():
            snapshot = env.snapshot()
            env.set_eval(True)
            obs, _ = self.env.reset()
            obs = obs.to(self.device)
            while (counts < targets).any():
                actions = policy(obs, stochastic_output=not deterministic)
                obs, rew, dones, extras = self.env.step(actions.to(self.env.device))
                obs = obs.to(self.device)
                cur_rew += rew.cpu()
                cur_len += 1
                outcome = extras["done_info"]["outcome"].cpu()
                for i in dones.nonzero().view(-1).cpu().tolist():
                    if counts[i] < targets[i]:
                        rewards.append(float(cur_rew[i]))
                        lengths.append(int(cur_len[i]))
                        successes.append(bool(outcome[i] == 0))
                        counts[i] += 1
                    cur_rew[i] = 0.0
                    cur_len[i] = 0
            env.set_eval(False)
            env.restore(snapshot)
        self.alg.train_mode()
        return np.array(rewards), np.array(lengths), np.array(successes)

    def _run_eval(self) -> None:
        t0 = time.time()
        rewards, lengths, successes = self.evaluate(self.n_eval_episodes)
        mean_r, std_r = float(rewards.mean()), float(rewards.std())
        self.eval_log["timesteps"].append(self.num_timesteps)
        self.eval_log["results"].append(rewards)
        self.eval_log["ep_lengths"].append(lengths)
        self.eval_log["successes"].append(successes)
        if self.logger.writer is not None:
            self.logger.writer.add_scalar("eval/mean_reward", mean_r, self.num_timesteps)
            self.logger.writer.add_scalar("eval/mean_ep_length", float(lengths.mean()), self.num_timesteps)
            self.logger.writer.add_scalar("eval/success_rate", float(successes.mean()), self.num_timesteps)
            np.savez(os.path.join(self.logger.log_dir, "evaluations.npz"),
                     **{k: np.array(v) for k, v in self.eval_log.items()})
        best = ""
        if mean_r > self.best_mean_reward:
            self.best_mean_reward = mean_r
            if self.models_dir:
                self.save(os.path.join(self.models_dir, "best_model.pt"))
            best = "  -> new best model"
        print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_r:.2f} +/- {std_r:.2f}, "
              f"success {100 * successes.mean():.0f}%, episode length {lengths.mean():.0f} "
              f"({time.time() - t0:.0f} s){best}", flush=True)

    # ------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        steps_per_it = self.cfg["num_steps_per_env"] * self.env.num_envs
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)
                    self._record_episodes(extras)
                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it
            prev_steps = self.num_timesteps
            self.num_timesteps += steps_per_it
            self.logger.log(it=it, start_it=start_it, total_it=total_it, collect_time=collect_time,
                            learn_time=learn_time, loss_dict=loss_dict, learning_rate=self.alg.learning_rate,
                            action_std=self.alg.get_policy().output_std, rnd_weight=None)
            self._log_pipe_stats(it)
            if self.models_dir and self.checkpoint_freq > 0 and \
                    self.num_timesteps // self.checkpoint_freq > prev_steps // self.checkpoint_freq:
                self.save(os.path.join(self.models_dir, f"model_{self.num_timesteps}_steps.pt"))
            if self.eval_freq > 0 and self.num_timesteps // self.eval_freq > prev_steps // self.eval_freq:
                self._run_eval()
        if self.models_dir:
            self.save(os.path.join(self.models_dir, "final_model.pt"))
        if self.logger.writer is not None:
            self.logger.stop_logging_writer()
