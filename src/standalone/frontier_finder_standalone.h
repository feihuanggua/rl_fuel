#ifndef FRONTIER_FINDER_STANDALONE_H
#define FRONTIER_FINDER_STANDALONE_H

#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <memory>
#include <vector>
#include <list>
#include <queue>
#include <algorithm>
#include <cmath>
#include <utility>

#include "edt_environment_standalone.h"
#include "perception_utils_standalone.h"
#include "raycast_standalone.h"
#include "params.h"

namespace fuel_rl {

struct Viewpoint {
  Eigen::Vector3d pos_;
  double yaw_;
  int visib_num_;
};

struct Frontier {
  std::vector<Eigen::Vector3d> cells_;
  std::vector<Eigen::Vector3d> filtered_cells_;
  Eigen::Vector3d average_;
  int id_;
  std::vector<Viewpoint> viewpoints_;
  Eigen::Vector3d box_min_, box_max_;
};

struct FrontierInfo {
  int id;
  std::vector<Eigen::Vector3d> cells;
  Eigen::Vector3d average;
  Eigen::Vector3d box_min, box_max;
  Eigen::Vector3d best_viewpoint_pos;
  double best_viewpoint_yaw;
  int best_viewpoint_visib_num;
  int frontier_size;
};

class FrontierFinder {
public:
  using Ptr = std::shared_ptr<FrontierFinder>;

  FrontierFinder(const EDTEnvironment::Ptr& edt, const FrontierParams& fparams, const PerceptionParams& pparams);
  ~FrontierFinder();

  void searchFrontiers();
  void computeFrontiersToVisit();

  void getFrontiers(std::vector<std::vector<Eigen::Vector3d>>& clusters);
  void getFrontierBoxes(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes);
  void getTopViewpointsInfo(const Eigen::Vector3d& cur_pos, std::vector<Eigen::Vector3d>& points,
                            std::vector<double>& yaws, std::vector<Eigen::Vector3d>& averages);

  // RL-specific: get structured frontier info
  std::vector<FrontierInfo> getFrontierInfos(const Eigen::Vector3d& agent_pos);

  bool isFrontierCovered();
  void wrapYaw(double& yaw);

  int countVisibleCells(const Eigen::Vector3d& pos, const double& yaw,
                        const std::vector<Eigen::Vector3d>& cluster);

  PerceptionUtils::Ptr percep_utils_;

private:
  void splitLargeFrontiers(std::list<Frontier>& frontiers);
  bool splitHorizontally(const Frontier& frontier, std::list<Frontier>& splits);
  bool isFrontierChanged(const Frontier& ft);
  bool haveOverlap(const Eigen::Vector3d& min1, const Eigen::Vector3d& max1,
                   const Eigen::Vector3d& min2, const Eigen::Vector3d& max2);
  void computeFrontierInfo(Frontier& frontier);
  void downsample(const std::vector<Eigen::Vector3d>& cluster_in, std::vector<Eigen::Vector3d>& cluster_out);
  void sampleViewpoints(Frontier& frontier);
  bool isNearUnknown(const Eigen::Vector3d& pos);
  std::vector<Eigen::Vector3i> sixNeighbors(const Eigen::Vector3i& voxel);
  std::vector<Eigen::Vector3i> allNeighbors(const Eigen::Vector3i& voxel);
  bool isNeighborUnknown(const Eigen::Vector3i& voxel);
  void expandFrontier(const Eigen::Vector3i& first);

  int toadr(const Eigen::Vector3i& idx);
  bool knownfree(const Eigen::Vector3i& idx);
  bool inmap(const Eigen::Vector3i& idx);

  std::vector<char> frontier_flag_;
  std::list<Frontier> frontiers_, dormant_frontiers_, tmp_frontiers_;
  std::list<Frontier>::iterator first_new_ftr_;

  int cluster_min_;
  double cluster_size_xy_, cluster_size_z_;
  double candidate_rmax_, candidate_rmin_, candidate_dphi_, min_candidate_dist_;
  double min_candidate_clearance_;
  int down_sample_;
  double min_view_finish_fraction_, resolution_;
  int min_visib_num_, candidate_rnum_;

  EDTEnvironment::Ptr edt_env_;
  std::unique_ptr<RayCaster> raycaster_;
};

}  // namespace fuel_rl

#endif  // FRONTIER_FINDER_STANDALONE_H
