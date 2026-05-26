"""Training script for FUEL RL environment using Stable-Baselines3."""
import argparse
import os
import sys

import numpy as np


def make_env(map_paths=None, map_size=(20, 20, 3), num_pillars=15,
             max_steps=100, single_frontier=True):
    """Create environment factory."""
    import gymnasium as gym
    from fuel_rl.env import FuelRLEnv, FuelRLEnvSingleFrontier
    cls = FuelRLEnvSingleFrontier if single_frontier else FuelRLEnv
    env = cls(
        map_paths=map_paths,
        map_size=map_size,
        num_pillars=num_pillars,
        max_steps=max_steps,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    return env


def train(args):
    """Train RL agent."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    # Find available PCD files
    from fuel_rl.map_loader import pcd_file_list
    map_paths = pcd_file_list()
    if not map_paths:
        print("No PCD files found, using random map generation")
        map_paths = None

    print(f"Map paths: {map_paths}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Policy: {args.policy}")
    print(f"Single frontier mode: {args.single_frontier}")

    # Create train env
    train_env = make_env(
        map_paths=map_paths,
        map_size=(args.map_size, args.map_size, args.map_height),
        num_pillars=args.num_pillars,
        max_steps=args.max_steps,
        single_frontier=args.single_frontier,
    )

    # Create eval env
    eval_env = make_env(
        map_paths=map_paths[:1] if map_paths else None,
        map_size=(args.map_size, args.map_size, args.map_height),
        num_pillars=args.num_pillars,
        max_steps=args.max_steps,
        single_frontier=args.single_frontier,
    )

    # Create model
    tb_log = args.log_dir if args.use_tensorboard else None
    model = PPO(
        args.policy,
        train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        tensorboard_log=tb_log,
    )

    # Callbacks
    callbacks = []

    # Metrics logging callback
    from fuel_rl.callbacks import make_sb3_callback
    metrics_cb, _ = make_sb3_callback(log_dir=args.log_dir)
    callbacks.append(metrics_cb)

    if args.eval_freq > 0:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.log_dir, "best_model"),
            log_path=os.path.join(args.log_dir, "eval_logs"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            warn=False,
        )
        callbacks.append(eval_cb)

    if args.checkpoint_freq > 0:
        ckpt_cb = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=os.path.join(args.log_dir, "checkpoints"),
            name_prefix="fuel_ppo",
        )
        callbacks.append(ckpt_cb)

    os.makedirs(args.log_dir, exist_ok=True)

    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )

    # Save final model
    final_path = os.path.join(args.log_dir, "fuel_ppo_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")

    train_env.close()
    eval_env.close()

    # Generate plots
    print("\nGenerating training plots...")
    from fuel_rl.plot_metrics import plot_metrics
    plot_metrics(
        os.path.join(args.log_dir, "metrics.json"),
        output_dir=args.log_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Train FUEL RL agent")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--policy", type=str, default="MultiInputPolicy")
    parser.add_argument("--single-frontier", action="store_true", default=True,
                        help="Use single-frontier mode (auto-select nearest)")
    parser.add_argument("--no-single-frontier", dest="single_frontier", action="store_false",
                        help="Use full exploration mode (select frontier + viewpoint)")

    # Environment
    parser.add_argument("--map-size", type=float, default=20.0)
    parser.add_argument("--map-height", type=float, default=3.0)
    parser.add_argument("--num-pillars", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=100)

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Logging
    parser.add_argument("--log-dir", type=str, default="./fuel_rl_tensorboard/")
    parser.add_argument("--use-tensorboard", action="store_true", default=False,
                        help="Enable TensorBoard logging (requires tensorboard package)")
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=10000)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
