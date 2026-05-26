"""Quick test script for FUEL RL environment with visualization."""
import numpy as np
import sys
sys.path.insert(0, '/home/jd3/FUEL/rl_fuel')

from fuel_rl.env import FuelRLEnv, FuelRLEnvSingleFrontier
from fuel_rl.visualizer import FuelVisualizer
from fuel_rl import FuelEnvCore, SDFMapParams, FrontierParams, PerceptionParams, AstarParams
from fuel_rl.map_loader import generate_random_map_for_fuel


def test_core_and_viz():
    """Test core functionality and visualization."""
    print("=== Test 1: Core + Visualization ===")

    core = FuelEnvCore()
    mp = SDFMapParams()
    mp.map_size_x = 20.0; mp.map_size_y = 20.0; mp.map_size_z = 3.0
    mp.box_min_x = -9.0; mp.box_min_y = -9.0; mp.box_min_z = 0.0
    mp.box_max_x = 9.0;  mp.box_max_y = 9.0;  mp.box_max_z = 2.8
    fp = FrontierParams(); fp.cluster_min = 10
    core.init(mp, fp, PerceptionParams(), AstarParams())

    # Random map
    rng = np.random.default_rng(42)
    obstacles = []
    for _ in range(15):
        cx, cy = rng.uniform(-7, 7), rng.uniform(-7, 7)
        if abs(cx) < 2 and abs(cy) < 2: continue
        w = rng.uniform(0.4, 0.7)
        for dx in np.arange(-w/2, w/2, 0.1):
            for dy in np.arange(-w/2, w/2, 0.1):
                for dz in np.arange(0, 2.0, 0.1):
                    obstacles.append([cx+dx, cy+dy, dz])
    obstacles = np.array(obstacles)
    core.load_map_from_points(obstacles)
    core.reset_map()

    # Initial observation
    start = np.array([0.0, 0.0, 1.5])
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(start, yaw)

    frontiers = core.detect_frontiers(start)
    print(f"Frontiers: {len(frontiers)}")
    print(f"Progress: {core.get_exploration_progress():.4f}")

    # Plan and visit first frontier
    executed = [start.copy()]
    planned = []
    if frontiers:
        f = frontiers[0]
        goal = np.array(f.best_viewpoint_pos)
        path = core.plan_path(start, goal)
        planned = [np.array(p) for p in path]
        if path:
            core.simulate_observation(goal, f.best_viewpoint_yaw)
            executed.append(goal.copy())
            frontiers2 = core.detect_frontiers(goal)
            print(f"After visit: progress={core.get_exploration_progress():.4f}, frontiers={len(frontiers2)}")

    # Visualize
    viz = FuelVisualizer(slice_height=1.5, figsize=(10, 10))
    viz.render_2d(core, start, 0.0, frontiers=frontiers,
                  planned_path=planned, executed_path=executed,
                  title='FUEL RL Environment - Initial State',
                  save_path='/tmp/fuel_rl_test.png')
    print("Saved visualization to /tmp/fuel_rl_test.png")
    plt.close('all')


def test_gym_env():
    """Test Gymnasium environment."""
    print("\n=== Test 2: Gymnasium Environment ===")
    from matplotlib import pyplot as plt

    env = FuelRLEnvSingleFrontier(
        num_pillars=15,
        max_steps=50,
    )

    obs, info = env.reset()
    print(f"Obs keys: {list(obs.keys())}")
    print(f"Global: {obs['global']}")
    print(f"Mask sum: {obs['mask'].sum()}")
    print(f"Frontier obs shape: {obs['frontiers'].shape}")

    total_reward = 0
    viz = FuelVisualizer(slice_height=1.5, figsize=(10, 10))
    frames = []

    for step in range(20):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0 or terminated:
            print(f"Step {step}: reward={reward:.3f}, progress={info.get('exploration_progress', 0):.4f}, "
                  f"frontiers={info.get('n_frontiers', 0)}, error={info.get('error', 'none')}")

            # Save frame
            viz.render_2d(env.core, env.agent_pos, env.agent_yaw,
                         frontiers=env.frontiers,
                         planned_path=env.planned_path,
                         executed_path=env.executed_path,
                         title=f'Step {step} | Reward: {total_reward:.1f}',
                         save_path=f'/tmp/fuel_rl_step_{step:03d}.png')
            plt.close('all')

        if terminated or truncated:
            print(f"Done at step {step}. Terminated={terminated}, Truncated={truncated}")
            break

    print(f"Total reward: {total_reward:.2f}")
    print("Env test PASSED")


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    test_core_and_viz()
    test_gym_env()
