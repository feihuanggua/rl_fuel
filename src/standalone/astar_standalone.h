#ifndef ASTAR_STANDALONE_H
#define ASTAR_STANDALONE_H

#include <Eigen/Eigen>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <queue>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <chrono>

#include "edt_environment_standalone.h"
#include "params.h"

namespace fuel_rl {

struct matrix_hash {
  std::size_t operator()(Eigen::Vector3i const& idx) const {
    size_t seed = 0;
    for (int i = 0; i < 3; ++i) {
      seed ^= std::hash<int>()(idx[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

struct AstarNode {
  Eigen::Vector3i index;
  Eigen::Vector3d position;
  double g_score, f_score;
  AstarNode* parent;

  AstarNode() { parent = nullptr; }
};

struct AstarNodeComparator {
  bool operator()(AstarNode* a, AstarNode* b) { return a->f_score > b->f_score; }
};

class Astar {
public:
  using Ptr = std::shared_ptr<Astar>;
  enum { REACH_END = 1, NO_PATH = 2 };

  Astar();
  ~Astar();

  void init(const AstarParams& params, const EDTEnvironment::Ptr& env);
  void reset();
  int search(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt);
  void setResolution(const double& res);
  static double pathLength(const std::vector<Eigen::Vector3d>& path);

  std::vector<Eigen::Vector3d> getPath();
  std::vector<Eigen::Vector3d> getVisited();
  double getEarlyTerminateCost();

private:
  void backtrack(const AstarNode* end_node, const Eigen::Vector3d& end);
  void posToIndex(const Eigen::Vector3d& pt, Eigen::Vector3i& idx);
  double getDiagHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);

  std::vector<AstarNode*> path_node_pool_;
  int use_node_num_, iter_num_;
  std::priority_queue<AstarNode*, std::vector<AstarNode*>, AstarNodeComparator> open_set_;
  std::unordered_map<Eigen::Vector3i, AstarNode*, matrix_hash> open_set_map_;
  std::unordered_map<Eigen::Vector3i, int, matrix_hash> close_set_map_;
  std::vector<Eigen::Vector3d> path_nodes_;
  double early_terminate_cost_;

  EDTEnvironment::Ptr edt_env_;

  double margin_;
  int allocate_num_;
  double tie_breaker_;
  double resolution_, inv_resolution_;
  double lambda_heu_;
  double max_search_time_;
  Eigen::Vector3d map_size_3d_, origin_;
};

}  // namespace fuel_rl

#endif  // ASTAR_STANDALONE_H
