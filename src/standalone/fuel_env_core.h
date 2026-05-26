#ifndef FUEL_ENV_CORE_H
#define FUEL_ENV_CORE_H

#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <memory>

#include "params.h"
#include "sdf_map_standalone.h"
#include "edt_environment_standalone.h"
#include "perception_utils_standalone.h"
#include "astar_standalone.h"
#include "frontier_finder_standalone.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>

namespace fuel_rl {

class FuelEnvCore {
public:
  using Ptr = std::shared_ptr<FuelEnvCore>;

  FuelEnvCore();
  ~FuelEnvCore();

  void init(const SDFMapParams& map_params, const FrontierParams& frontier_params,
            const PerceptionParams& perception_params, const AstarParams& astar_params);

  // Map loading
  void loadMapFromPCD(const std::string& pcd_path);
  void loadMapFromPoints(const std::vector<Eigen::Vector3d>& obstacle_points);
  void resetMap();

  // Simulate sensor observation at a viewpoint
  void simulateObservation(const Eigen::Vector3d& camera_pos, double yaw);

  // Frontier detection
  std::vector<FrontierInfo> detectFrontiers(const Eigen::Vector3d& agent_pos);

  // Path planning
  std::vector<Eigen::Vector3d> planPath(const Eigen::Vector3d& start, const Eigen::Vector3d& goal);
  double computePathCost(const Eigen::Vector3d& start, const Eigen::Vector3d& goal);

  // Visibility checking
  int countVisibleCells(const Eigen::Vector3d& pos, double yaw,
                        const std::vector<Eigen::Vector3d>& cells);

  // Map queries
  int getOccupancy(const Eigen::Vector3d& pos);
  double getDistance(const Eigen::Vector3d& pos);
  double getResolution();
  void getRegion(Eigen::Vector3d& origin, Eigen::Vector3d& size);
  void getBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax);

  // Statistics
  int countUnknownVoxels();
  int countFreeVoxels();
  int countOccupiedVoxels();
  double getExplorationProgress();

  // Local voxel grid for observation
  std::vector<int8_t> getLocalVoxelGrid(const Eigen::Vector3d& center, int grid_size);

  // 2D slice for visualization
  std::vector<int8_t> getOccupancySlice2D(double height);

  // Get map dimensions (nx, ny, nz)
  Eigen::Vector3i getMapVoxelNum();

  // Ground plane: fill ground-level voxels as occupied
  void setGroundPlane(bool enable, double height = 0.0);

private:
  SDFMap::Ptr sdf_map_;
  EDTEnvironment::Ptr edt_env_;
  PerceptionUtils::Ptr perception_;
  Astar::Ptr astar_;
  FrontierFinder::Ptr frontier_finder_;

  // Ground truth map for sensor simulation
  pcl::PointCloud<pcl::PointXYZ>::Ptr gt_cloud_;
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr gt_kdtree_;

  // Ground plane state
  bool ground_plane_enabled_ = false;
  double ground_plane_height_ = 0.0;
};

}  // namespace fuel_rl

#endif  // FUEL_ENV_CORE_H
