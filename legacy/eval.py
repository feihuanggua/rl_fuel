"""Evaluation and visualization script for trained FUEL RL agent."""
import argparse
import os
import glob
import numpy as np


def evaluate(args):
    """Run trained agent on environment with visualization."""
    from matplotlib import pyplot as plt
    from fuel_rl.env import FuelRLEnv, FuelRLEnvSingleFrontier
    from fuel_rl.visualizer import FuelVisualizer
    from fuel_rl.map_loader import pcd_file_list

    # Find map
    map_paths = pcd_file_list()
    if not map_paths:
        print("No PCD files found, using random map")
        map_paths = None

    # Create env
    cls = FuelRLEnvSingleFrontier if args.single_frontier else FuelRLEnv
    env = cls(
        map_paths=map_paths,
        map_size=(args.map_size, args.map_size, args.map_height),
        num_pillars=args.num_pillars,
        max_steps=args.max_steps,
        visualize=False,
    )

    # Load model
    from stable_baselines3 import PPO
    model = PPO.load(args.model_path)

    viz = FuelVisualizer(slice_height=1.5, figsize=(12, 12))
    frames_dir = args.output_dir
    os.makedirs(frames_dir, exist_ok=True)

    # Run episode
    obs, _ = env.reset(options={"map_path": args.map_path} if args.map_path else {})
    done = False
    step = 0
    total_reward = 0
    frame_paths = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Save visualization frame
        frame_path = os.path.join(frames_dir, f"step_{step:04d}.png")
        viz.render_2d(
            env.core, env.agent_pos, env.agent_yaw,
            frontiers=env.frontiers,
            planned_path=env.planned_path,
            executed_path=env.executed_path,
            title=f'Step {step} | Reward: {total_reward:.1f} | Progress: {info.get("exploration_progress", 0):.1%}',
            save_path=frame_path,
        )
        plt.close('all')
        frame_paths.append(frame_path)

        if step % 5 == 0 or done:
            error = info.get('error', 'none')
            print(f"Step {step:3d}: reward={reward:+7.2f} total={total_reward:+8.2f} "
                  f"progress={info.get('exploration_progress', 0):.4f} "
                  f"frontiers={info.get('n_frontiers', '?')} error={error}")

        step += 1
        if step >= args.max_steps:
            break

    print(f"\nEpisode finished: {step} steps, total reward: {total_reward:.2f}")

    # Generate animation
    if args.animate and frame_paths:
        try:
            import imageio
            images = [imageio.imread(p) for p in frame_paths]
            anim_path = os.path.join(args.output_dir, "exploration.gif")
            imageio.mimsave(anim_path, images, fps=5)
            print(f"Animation saved to {anim_path}")
        except ImportError:
            print("imageio not installed, skipping animation")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate FUEL RL agent")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model (.zip)")
    parser.add_argument("--map-path", type=str, default=None,
                        help="Specific PCD map file to evaluate on")
    parser.add_argument("--output-dir", type=str, default="/tmp/fuel_rl_eval")
    parser.add_argument("--animate", action="store_true",
                        help="Generate GIF animation")
    parser.add_argument("--single-frontier", action="store_true", default=True)
    parser.add_argument("--map-size", type=float, default=20.0)
    parser.add_argument("--map-height", type=float, default=3.0)
    parser.add_argument("--num-pillars", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=200)

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
