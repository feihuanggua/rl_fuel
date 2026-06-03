#include "fuel_env_core.h"
#include <iostream>
#include <chrono>

namespace fuel_rl {

static auto now() { return std::chrono::steady_clock::now(); }
static double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
static double elapsed_ms(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) {
  return std::chrono::duration<double, std::milli>(to - from).count();
}

FuelEnvCore::FuelEnvCore() {}

FuelEnvCore::~FuelEnvCore() {}

void FuelEnvCore::init(const SDFMapParams& map_params, const FrontierParams& frontier_params,
                       const PerceptionParams& perception_params, const AstarParams& astar_params) {
  // Create and init SDF map
  sdf_map_.reset(new SDFMap);
  sdf_map_->init(map_params);

  // Create and init EDT environment
  edt_env_.reset(new EDTEnvironment);
  edt_env_->init();
  edt_env_->setMap(sdf_map_);

  // Create and init perception utils
  perception_.reset(new PerceptionUtils);
  perception_->init(perception_params);

  // Create and init A*
  astar_.reset(new Astar);
  astar_->init(astar_params, edt_env_);

  // Create and init frontier finder
  frontier_finder_.reset(new FrontierFinder(edt_env_, frontier_params, perception_params));

  // Init ground truth storage
  gt_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  gt_kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>);
}

void FuelEnvCore::loadMapFromPCD(const std::string& pcd_path) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) == -1) {
    std::cerr << "Failed to load PCD file: " << pcd_path << std::endl;
    return;
  }
  gt_cloud_ = cloud;
  gt_kdtree_->setInputCloud(gt_cloud_);

  // Convert to vector and inject obstacles
  std::vector<Eigen::Vector3d> points;
  points.reserve(cloud->size());
  for (auto& pt : cloud->points) {
    points.push_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
  }
  loadMapFromPoints(points);
}

void FuelEnvCore::loadMapFromPoints(const std::vector<Eigen::Vector3d>& obstacle_points) {
  // Store as ground truth cloud for sensor simulation (KdTree only)
  gt_cloud_->clear();
  gt_cloud_->reserve(obstacle_points.size());
  for (auto& pt : obstacle_points) {
    gt_cloud_->push_back(pcl::PointXYZ(pt[0], pt[1], pt[2]));
  }
  gt_kdtree_->setInputCloud(gt_cloud_);

  // Inject obstacles into belief SDF map so they appear as solid known obstacles
  sdf_map_->injectObstacles(obstacle_points);
}

void FuelEnvCore::resetMap() {
  // Reset belief map to all unknown
  auto& md = sdf_map_->md();
  auto& mp = sdf_map_->mp();
  int buffer_size = mp->map_voxel_num_(0) * mp->map_voxel_num_(1) * mp->map_voxel_num_(2);
  for (int i = 0; i < buffer_size; ++i) {
    md->occupancy_buffer_[i] = mp->clamp_min_log_ - mp->unknown_flag_;
    md->occupancy_buffer_inflate_[i] = 0;
    md->distance_buffer_[i] = mp->default_dist_;
  }
  md->local_bound_min_ = Eigen::Vector3i::Zero();
  md->local_bound_max_ = mp->map_voxel_num_ - Eigen::Vector3i::Ones();
  md->reset_updated_box_ = true;
  md->local_updated_ = false;

  // Re-inject ground-truth obstacles into belief map as known occupied cells
  if (gt_cloud_ && !gt_cloud_->empty()) {
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(gt_cloud_->size());
    for (auto& p : *gt_cloud_) pts.push_back(Eigen::Vector3d(p.x, p.y, p.z));
    sdf_map_->injectObstacles(pts);
  }

  // Apply ground plane if enabled
  if (ground_plane_enabled_) {
    sdf_map_->addGroundPlane(ground_plane_height_);
  }
}

void FuelEnvCore::setGroundPlane(bool enable, double height) {
  ground_plane_enabled_ = enable;
  ground_plane_height_ = height;
  // Ground is applied during resetMap(), not immediately, so it survives reloads
}

void FuelEnvCore::simulateObservation(const Eigen::Vector3d& camera_pos, double yaw) {
  auto t0 = now();
  // FUEL pinhole camera parameters (matching camera.yaml and exploration.launch)
  const int width = 640, height = 480;
  const double fx = 387.229, fy = 387.229, cx = 321.046, cy = 243.449;
  const int skip_pixel = perception_->getSkipPixel();  // from params
  const int margin = 2;
  const double max_range = perception_->getMaxDist();  // 4.5m
  const double min_range = 0.2;

  // Body -> World rotation (body: x-forward, y-left, z-up)
  Eigen::Matrix3d R_wb;
  R_wb << cos(yaw), -sin(yaw), 0.0,
          sin(yaw),  cos(yaw), 0.0,
          0.0,       0.0,      1.0;

  // Camera -> Body transform (from pcl_render_node.cpp)
  // cam02body: camera looks forward from UAV
  // Camera X = Body Z, Camera Y = -Body X, Camera Z = -Body Y
  Eigen::Matrix3d R_bc;
  R_bc << 0.0,  0.0,  1.0,
         -1.0,  0.0,  0.0,
          0.0, -1.0,  0.0;
  Eigen::Matrix3d R_wc = R_wb * R_bc;

  // Camera position in world (body frame origin)
  // In FUEL, cam02body translation is [0,0,0]
  Eigen::Vector3d cam_pos_world = camera_pos;

  std::vector<Eigen::Vector3d> hit_points;
  // Reserve: (width-2*margin)/skip_pixel * (height-2*margin)/skip_pixel
  int n_u = (width - 2 * margin) / skip_pixel;
  int n_v = (height - 2 * margin) / skip_pixel;
  hit_points.reserve(n_u * n_v);

  double resolution = sdf_map_->getResolution();

  int ray_idx = 0;
  auto deadline = now() + std::chrono::seconds(10);
  bool timeout = false;
  for (int v = margin; v < height - margin && !timeout; v += skip_pixel) {
    for (int u = margin; u < width - margin && !timeout; u += skip_pixel) {
      ray_idx++;
      Eigen::Vector3d dir_cam((u - cx) / fx, (v - cy) / fy, 1.0);
      dir_cam.normalize();
      Eigen::Vector3d ray_dir = R_wc * dir_cam;

      double step_size = resolution * 2.0;
      bool hit = false;
      for (double t = step_size; t < max_range; t += step_size) {
        Eigen::Vector3d sample = cam_pos_world + t * ray_dir;
        pcl::PointXYZ query(sample[0], sample[1], sample[2]);

        std::vector<int> indices(1);
        std::vector<float> distances(1);
        if (gt_kdtree_->radiusSearch(query, resolution * 1.5, indices, distances) > 0) {
          if (t >= min_range) {
            hit_points.push_back(sample);
          }
          hit = true;
          break;
        }
      }
      if (!hit) {
        static const double depth_filter_maxdist = 5.0;
        Eigen::Vector3d free_pt = cam_pos_world + depth_filter_maxdist * ray_dir;
        hit_points.push_back(free_pt);
      }
      if (ray_idx % 2000 == 0 && now() > deadline) {
        std::cerr << "[sim_obs TIMEOUT] ray " << ray_idx << " exceeded 10s, aborting rays" << std::endl;
        timeout = true;
      }
    }
  }

  if (!hit_points.empty()) {
    auto t1 = now();
    sdf_map_->inputPointCloud(hit_points, camera_pos);
    auto t2 = now();
    sdf_map_->clearAndInflateLocalMap();
    auto t3 = now();
    sdf_map_->updateESDF3d();
    auto t4 = now();
    double total = ms_since(t0);
    double input_pc = elapsed_ms(t1, t2);
    double inflate = elapsed_ms(t2, t3);
    double esdf = elapsed_ms(t3, t4);
    if (total > 1000.0) {
      std::cerr << "[SLOW sim_obs >1s] total=" << total << "ms raycast=" << elapsed_ms(t0, t1)
                << " inputPC=" << input_pc << " inflate=" << inflate << " esdf=" << esdf << std::endl;
    }
  }
}

void FuelEnvCore::inputHitPoints(const std::vector<Eigen::Vector3d>& hit_points,
                                  const Eigen::Vector3d& camera_pos) {
  if (hit_points.empty()) return;
  sdf_map_->inputPointCloud(hit_points, camera_pos);
  sdf_map_->clearAndInflateLocalMap();
  sdf_map_->updateESDF3d();
}

std::vector<FrontierInfo> FuelEnvCore::detectFrontiers(const Eigen::Vector3d& agent_pos) {
  auto t0 = now();
  frontier_finder_->searchFrontiers();
  auto t1 = now();
  frontier_finder_->computeFrontiersToVisit();
  auto t2 = now();
  auto infos = frontier_finder_->getFrontierInfos(agent_pos);
  auto t3 = now();
  double total = ms_since(t0);
  if (total > 500.0) {
    std::cerr << "[SLOW detect] total=" << total << "ms search=" << elapsed_ms(t0, t1)
              << " compute=" << elapsed_ms(t1, t2) << " getInfo=" << elapsed_ms(t2, t3)
              << " nf=" << infos.size() << std::endl;
  }
  return infos;
}

std::vector<Eigen::Vector3d> FuelEnvCore::planPath(const Eigen::Vector3d& start,
                                                     const Eigen::Vector3d& goal) {
  astar_->reset();
  int result = astar_->search(start, goal);
  if (result == Astar::REACH_END) {
    return astar_->getPath();
  }
  return {};
}

double FuelEnvCore::computePathCost(const Eigen::Vector3d& start, const Eigen::Vector3d& goal) {
  auto path = planPath(start, goal);
  return Astar::pathLength(path);
}

int FuelEnvCore::countVisibleCells(const Eigen::Vector3d& pos, double yaw,
                                    const std::vector<Eigen::Vector3d>& cells) {
  return frontier_finder_->countVisibleCells(pos, yaw, cells);
}

int FuelEnvCore::getOccupancy(const Eigen::Vector3d& pos) { return sdf_map_->getOccupancy(pos); }

double FuelEnvCore::getDistance(const Eigen::Vector3d& pos) { return sdf_map_->getDistance(pos); }

double FuelEnvCore::getResolution() { return sdf_map_->getResolution(); }

void FuelEnvCore::getRegion(Eigen::Vector3d& origin, Eigen::Vector3d& size) { sdf_map_->getRegion(origin, size); }

void FuelEnvCore::getBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax) { sdf_map_->getBox(bmin, bmax); }

int FuelEnvCore::countUnknownVoxels() { return sdf_map_->countUnknownVoxels(); }
int FuelEnvCore::countFreeVoxels() { return sdf_map_->countFreeVoxels(); }
int FuelEnvCore::countOccupiedVoxels() { return sdf_map_->countOccupiedVoxels(); }
double FuelEnvCore::getExplorationProgress() { return sdf_map_->getExplorationProgress(); }

std::vector<int8_t> FuelEnvCore::getLocalVoxelGrid(const Eigen::Vector3d& center, int grid_size) {
  return sdf_map_->getLocalVoxelGrid(center, grid_size);
}

std::vector<int8_t> FuelEnvCore::getOccupancySlice2D(double height) {
  return sdf_map_->getOccupancySlice2D(height);
}

Eigen::Vector3i FuelEnvCore::getMapVoxelNum() { return sdf_map_->mp()->map_voxel_num_; }

}  // namespace fuel_rl
