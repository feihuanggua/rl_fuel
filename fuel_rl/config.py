"""全局配置."""
import torch

# --- 设备 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 体素网格 ---
GRID_SIZE = 32          # XY 方向网格大小
GRID_Z = 10             # Z 方向网格大小
VOXEL_RES = 0.2         # 体素分辨率 (m)
ROI_HALF = GRID_SIZE * VOXEL_RES / 2  # 3.2m 半范围

# --- 3通道定义 ---
CH_OCC = 0              # 障碍物
CH_FRONTIER = 1         # 前沿
CH_FREE = 2             # 自由空间

# --- 专家数据收集 ---
COLLECT_NUM_MAPS = 200
COLLECT_MAP_SIZE = (20.0, 20.0, 3.0)
COLLECT_NUM_PILLARS = 15
COLLECT_SAVE_PATH = "./fuel_rl_data/expert_data.pt"

# --- BC 训练 ---
BC_BATCH_SIZE = 128
BC_LR = 1e-3
BC_WEIGHT_DECAY = 1e-4
BC_EPOCHS = 100
BC_VAL_SPLIT = 0.1
BC_SAVE_DIR = "./fuel_rl_checkpoints/bc"

# --- PPO 微调 ---
PPO_LR_ACTOR = 1e-5        # lower to prevent drift from BC
PPO_LR_CRITIC = 5e-5
PPO_LR_BACKBONE = 1e-5
PPO_GAMMA = 0.99
PPO_EPS_CLIP = 0.15
PPO_K_EPOCHS = 2
PPO_UPDATE_TIMESTEP = 1000   # was 2000, more frequent updates
PPO_MAX_EPISODES = 50000
PPO_SAVE_DIR = "./fuel_rl_checkpoints/ppo_v3"
PPO_BC_CKPT = "./fuel_rl_checkpoints/bc_v3/best_model.pth"

# --- 编码器 ---
ENCODER_CHANNELS = [32, 64, 128]
ENCODER_EMBED_DIM = 512

# --- 视点头 ---
VIEWPOINT_POS_HIDDEN = 512
VIEWPOINT_YAW_HIDDEN = 256


# --- C++ 核心默认参数 ---
def default_map_params(
    size_x=20.0, size_y=20.0, size_z=3.0,
    box_min=None, box_max=None, resolution=0.1,
):
    from fuel_rl import SDFMapParams
    p = SDFMapParams()
    p.resolution = resolution
    p.map_size_x = size_x
    p.map_size_y = size_y
    p.map_size_z = size_z
    p.obstacles_inflation = 0.199
    p.ground_height = 0.0
    p.default_dist = 0.0
    p.p_hit = 0.65
    p.p_miss = 0.35
    p.max_ray_length = 3.0
    p.optimistic = False
    if box_min is not None:
        p.box_min_x, p.box_min_y, p.box_min_z = box_min
    if box_max is not None:
        p.box_max_x, p.box_max_y, p.box_max_z = box_max
    return p


def default_frontier_params():
    from fuel_rl import FrontierParams
    p = FrontierParams()
    p.cluster_size_xy = 0.5
    p.candidate_rmin = 0.8
    p.candidate_rmax = 1.5
    p.min_visib_num = 3
    return p


def default_perception_params():
    from fuel_rl import PerceptionParams
    p = PerceptionParams()
    p.skip_pixel = 2  # 2=FUEL default, 4=faster training
    return p

def fast_perception_params():
    from fuel_rl import PerceptionParams
    p = PerceptionParams()
    p.skip_pixel = 4  # ~19k rays per obs (vs ~75k FUEL default)
    return p


def default_astar_params():
    from fuel_rl import AstarParams
    return AstarParams()
