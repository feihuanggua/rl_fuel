#ifndef FUEL_RL_PARAMS_H
#define FUEL_RL_PARAMS_H

#include <cmath>

namespace fuel_rl {

struct SDFMapParams {
  double resolution = 0.1;
  double map_size_x = 50.0;
  double map_size_y = 50.0;
  double map_size_z = 5.0;
  double obstacles_inflation = 0.199;
  double local_bound_inflate = 0.5;
  int local_map_margin = 50;
  double ground_height = -1.0;
  double default_dist = 0.0;
  bool optimistic = false;
  bool signed_dist = false;
  double p_hit = 0.65;
  double p_miss = 0.35;
  double p_min = 0.12;
  double p_max = 0.90;
  double p_occ = 0.80;
  double max_ray_length = 4.5;
  double virtual_ceil_height = -10.0;
  double box_min_x = 0.0;  // 0 means use map boundary
  double box_min_y = 0.0;
  double box_min_z = 0.0;
  double box_max_x = 0.0;
  double box_max_y = 0.0;
  double box_max_z = 0.0;
};

struct FrontierParams {
  int cluster_min = 100;
  double cluster_size_xy = 2.0;
  double cluster_size_z = 10.0;
  double min_candidate_dist = 0.75;
  double min_candidate_clearance = 0.21;
  double candidate_dphi = 15.0 * M_PI / 180.0;
  int candidate_rnum = 3;
  double candidate_rmin = 1.5;
  double candidate_rmax = 2.5;
  int down_sample = 3;
  int min_visib_num = 15;
  double min_view_finish_fraction = 0.2;
};

struct PerceptionParams {
  double top_angle = 0.56125;
  double left_angle = 0.69222;
  double right_angle = 0.68901;
  double max_dist = 4.5;
  double vis_dist = 1.0;
  int skip_pixel = 2;  // depth image subsampling: 2=FUEL default, 4=faster
};

struct AstarParams {
  double lambda_heu = 10000.0;
  double resolution_astar = 0.2;
  double max_search_time = 0.001;
  int allocate_num = 1000000;
};

struct ExplorationParams {
  double vm = 1.0;
  double am = 1.0;
  double yd = 60.0 * M_PI / 180.0;
  double ydd = 90.0 * M_PI / 180.0;
  double w_dir = 1.5;
};

}  // namespace fuel_rl

#endif  // FUEL_RL_PARAMS_H
