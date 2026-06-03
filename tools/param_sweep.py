"""Quick parameter sweep to find config where Closest reaches ~95% coverage."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fuel_rl"))
import numpy as np
from fuel_rl import FuelEnvCore, SDFMapParams, FrontierParams
from fuel_rl.config import default_map_params, fast_perception_params, default_astar_params
from fuel_rl.map_loader import generate_random_map_for_fuel


def test_config(seed, max_ray, rmin, rmax, cluster_xy, min_visib, max_steps):
    core = FuelEnvCore()
    mp = default_map_params(size_x=20.0, size_y=20.0, size_z=3.0,
                            box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8))
    mp.max_ray_length = max_ray
    fp = FrontierParams()
    fp.candidate_rmin = rmin
    fp.candidate_rmax = rmax
    fp.cluster_size_xy = cluster_xy
    fp.min_visib_num = min_visib
    core.init(mp, fp, fast_perception_params(), default_astar_params())

    pts = generate_random_map_for_fuel(20.0, 20.0, 3.0, 15, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(agent_pos, yaw)

    fails = 0
    steps_used = 0
    for step in range(max_steps):
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            steps_used = step
            break
        # try all frontiers, skip failed ones
        moved = False
        for f in sorted(frontiers, key=lambda f: np.linalg.norm(np.array(f.average) - agent_pos)):
            vp = np.array(f.best_viewpoint_pos)
            vp[2] = np.clip(vp[2], 0.5, 2.6)
            occ = core.get_occupancy(vp)
            if occ != 1:
                fails += 1
                continue
            core.simulate_observation(vp, f.best_viewpoint_yaw)
            agent_pos = vp.copy()
            moved = True
            break
        if not moved:
            steps_used = step
            break
        steps_used = step + 1
        if core.get_exploration_progress() >= 0.95:
            break

    cov = core.get_exploration_progress()
    return cov, fails, steps_used


# Sweep: vary max_ray, candidate_r, cluster_xy, max_steps
configs = [
    # max_ray, rmin, rmax, cluster_xy, min_visib, max_steps
    (3.0, 0.5, 1.2, 0.5, 3, 500),
    (3.0, 0.5, 1.2, 0.8, 3, 500),
    (3.0, 0.5, 1.2, 1.0, 3, 500),
    (3.0, 0.5, 1.5, 0.8, 3, 600),
    (3.0, 0.8, 1.5, 0.5, 3, 600),
    (3.0, 0.8, 1.5, 0.8, 5, 600),
    (3.0, 0.8, 1.5, 1.0, 5, 600),
    (3.0, 0.6, 1.2, 0.8, 3, 500),
    (3.0, 0.6, 1.2, 1.0, 3, 500),
]

print(f"{'ray':>4s} {'rmin':>4s} {'rmax':>4s} {'cxy':>4s} {'vis':>4s} {'step':>5s} | {'cov':>6s} {'fail':>5s} {'used':>5s}")
print("-" * 55)

for cfg in configs:
    ray, rmin, rmax, cxy, vis, ms = cfg
    cov, fails, used = test_config(42, ray, rmin, rmax, cxy, vis, ms)
    mark = " <<<" if cov >= 0.90 else ""
    print(f"{ray:4.1f} {rmin:4.1f} {rmax:4.1f} {cxy:4.1f} {vis:4d} {ms:5d} | {cov:6.3f} {fails:5d} {used:5d}{mark}")
