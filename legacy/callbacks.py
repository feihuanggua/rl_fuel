"""Training callbacks for logging and visualization."""
import os
import json
import time
import numpy as np
from typing import Dict, Any


class MetricsLogger:
    """Records metrics to JSON file. Called from SB3 callback."""

    def __init__(self, log_dir: str = "./fuel_rl_tensorboard"):
        self.log_dir = log_dir
        self.metrics_path = os.path.join(log_dir, "metrics.json")
        self.metrics: Dict[str, list] = {
            "timesteps": [],
            "wall_time": [],
            "ep_rew_mean": [],
            "ep_len_mean": [],
            "loss": [],
            "policy_gradient_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
            "explained_variance": [],
            "learning_rate": [],
        }
        self._start_time: float = 0
        os.makedirs(log_dir, exist_ok=True)

    def record(self, total_timesteps: int, values: Dict[str, Any]):
        if self._start_time == 0:
            self._start_time = time.time()
        self.metrics["timesteps"].append(total_timesteps)
        self.metrics["wall_time"].append(time.time() - self._start_time)
        for key in self.metrics:
            if key in ("timesteps", "wall_time"):
                continue
            self.metrics[key].append(values.get(key))
        self._save()

    def _save(self):
        tmp = self.metrics_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.metrics, f, default=float)
        os.replace(tmp, self.metrics_path)


def make_sb3_callback(log_dir: str = "./fuel_rl_tensorboard"):
    """Create SB3 callback that records both rollout and train metrics."""
    from stable_baselines3.common.callbacks import BaseCallback

    logger = MetricsLogger(log_dir=log_dir)

    class _Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._last_rollout_metrics = {}

        def _on_training_start(self):
            logger._start_time = time.time()

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            # After rollout: capture ep_rew_mean etc. They will be printed after this call.
            # We stash them and record together with train metrics after the update.
            pass

        def _on_training_end(self) -> None:
            # SB3 calls this after model.learn() ends, not per-iteration.
            pass

    # We need a different approach: override the logger to intercept all writes.
    # Simplest: wrap the model's logger after creation.
    class _MetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._pending_rollout: Dict[str, Any] = {}
            self._iteration = 0

        def _on_training_start(self):
            logger._start_time = time.time()

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            # Read current logger values — rollout block just got written
            ntv = dict(self.model._logger.name_to_value)
            if "rollout/ep_rew_mean" in ntv:
                self._pending_rollout["ep_rew_mean"] = ntv["rollout/ep_rew_mean"]
            if "rollout/ep_len_mean" in ntv:
                self._pending_rollout["ep_len_mean"] = ntv["rollout/ep_len_mean"]

        def _on_training_end(self):
            pass

    # Better approach: patch into SB3's log output.
    # We'll use a simpler method — check after each learn iteration.
    # SB3 PPO calls _on_rollout_end before and after policy update.
    # The trick: the second _on_rollout_end call has train metrics available.

    class __MetricsCallback(BaseCallback):
        """Records metrics each PPO iteration."""
        def __init__(self):
            super().__init__()
            self._prev_rollout: Dict[str, Any] = {}
            self._initialized = False

        def _on_training_start(self):
            logger._start_time = time.time()

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            # Get episode reward directly from unwrapped env
            ep_rew = None
            try:
                all_rewards = []
                for env in self.training_env.envs:
                    base = env
                    while hasattr(base, 'env'):
                        base = base.env
                    if hasattr(base, 'episode_rewards') and base.episode_rewards:
                        all_rewards.extend(base.episode_rewards)
                        base.episode_rewards = []
                if all_rewards:
                    ep_rew = float(np.mean(all_rewards))
            except Exception:
                pass

            # Get train metrics from logger (still present from previous update)
            ntv = dict(self.model._logger.name_to_value)
            train = {}
            mapping = {
                "loss": "train/loss",
                "policy_gradient_loss": "train/policy_gradient_loss",
                "value_loss": "train/value_loss",
                "entropy_loss": "train/entropy_loss",
                "approx_kl": "train/approx_kl",
                "clip_fraction": "train/clip_fraction",
                "explained_variance": "train/explained_variance",
                "learning_rate": "train/learning_rate",
            }
            for our_key, sb3_key in mapping.items():
                if sb3_key in ntv:
                    train[our_key] = ntv[sb3_key]

            if not self._initialized:
                self._prev_rollout = {"ep_rew_mean": ep_rew, "ep_len_mean": None}
                self._initialized = True
                # Record first point if we have train data
                if train:
                    combined = {**self._prev_rollout, **train}
                    logger.record(self.num_timesteps, combined)
                return

            # Merge rollout + train
            combined = {
                "ep_rew_mean": ep_rew,
                "ep_len_mean": None,
                **train,
            }
            if any(v is not None for v in combined.values()):
                logger.record(self.num_timesteps, combined)

    return __MetricsCallback(), logger
