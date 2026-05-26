#ifndef PERCEPTION_UTILS_STANDALONE_H
#define PERCEPTION_UTILS_STANDALONE_H

#include <Eigen/Eigen>
#include <cmath>
#include <vector>
#include <memory>
#include "params.h"

namespace fuel_rl {

class PerceptionUtils {
public:
  using Ptr = std::shared_ptr<PerceptionUtils>;

  PerceptionUtils() {}
  ~PerceptionUtils() {}

  void init(const PerceptionParams& params);

  void setPose(const Eigen::Vector3d& pos, const double& yaw);
  void getFOV(std::vector<Eigen::Vector3d>& list1, std::vector<Eigen::Vector3d>& list2);
  bool insideFOV(const Eigen::Vector3d& point);
  void getFOVBoundingBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax);

  double getMaxDist() const { return max_dist_; }
  int getSkipPixel() const { return skip_pixel_; }

private:
  Eigen::Vector3d pos_;
  double yaw_;
  std::vector<Eigen::Vector3d> normals_;

  double left_angle_, right_angle_, top_angle_, max_dist_, vis_dist_;
  int skip_pixel_;
  Eigen::Vector3d n_top_, n_bottom_, n_left_, n_right_;
  Eigen::Matrix4d T_cb_, T_bc_;
  std::vector<Eigen::Vector3d> cam_vertices1_, cam_vertices2_;
};

}  // namespace fuel_rl

#endif  // PERCEPTION_UTILS_STANDALONE_H
