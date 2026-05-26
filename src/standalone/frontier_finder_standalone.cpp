#include "frontier_finder_standalone.h"
#include <unordered_map>

namespace fuel_rl {

FrontierFinder::FrontierFinder(const EDTEnvironment::Ptr& edt, const FrontierParams& fparams,
                               const PerceptionParams& pparams) {
  this->edt_env_ = edt;
  int voxel_num = edt->sdf_map_->getVoxelNum();
  frontier_flag_ = std::vector<char>(voxel_num, 0);

  cluster_min_ = fparams.cluster_min;
  cluster_size_xy_ = fparams.cluster_size_xy;
  cluster_size_z_ = fparams.cluster_size_z;
  min_candidate_dist_ = fparams.min_candidate_dist;
  min_candidate_clearance_ = fparams.min_candidate_clearance;
  candidate_dphi_ = fparams.candidate_dphi;
  candidate_rnum_ = fparams.candidate_rnum;
  candidate_rmin_ = fparams.candidate_rmin;
  candidate_rmax_ = fparams.candidate_rmax;
  down_sample_ = fparams.down_sample;
  min_visib_num_ = fparams.min_visib_num;
  min_view_finish_fraction_ = fparams.min_view_finish_fraction;

  raycaster_.reset(new RayCaster);
  resolution_ = edt_env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  percep_utils_.reset(new PerceptionUtils);
  percep_utils_->init(pparams);
}

FrontierFinder::~FrontierFinder() {}

void FrontierFinder::searchFrontiers() {
  tmp_frontiers_.clear();

  // Full rescan: clear all existing frontiers and search entire map
  for (auto iter = frontiers_.begin(); iter != frontiers_.end();) {
    Eigen::Vector3i idx;
    for (auto cell : iter->cells_) {
      edt_env_->sdf_map_->posToIndex(cell, idx);
      frontier_flag_[toadr(idx)] = 0;
    }
    iter = frontiers_.erase(iter);
  }
  for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
    Eigen::Vector3i idx;
    for (auto cell : iter->cells_) {
      edt_env_->sdf_map_->posToIndex(cell, idx);
      frontier_flag_[toadr(idx)] = 0;
    }
    iter = dormant_frontiers_.erase(iter);
  }

  // Search entire map bounding box
  Eigen::Vector3d search_min, search_max;
  edt_env_->sdf_map_->getBox(search_min, search_max);
  Eigen::Vector3d box_min, box_max;
  edt_env_->sdf_map_->getBox(box_min, box_max);
  for (int k = 0; k < 3; ++k) {
    search_min[k] = std::max(search_min[k], box_min[k]);
    search_max[k] = std::min(search_max[k], box_max[k]);
  }
  Eigen::Vector3i min_id, max_id;
  edt_env_->sdf_map_->posToIndex(search_min, min_id);
  edt_env_->sdf_map_->posToIndex(search_max, max_id);

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        Eigen::Vector3i cur(x, y, z);
        if (frontier_flag_[toadr(cur)] == 0 && knownfree(cur) && isNeighborUnknown(cur)) {
          expandFrontier(cur);
        }
      }
  splitLargeFrontiers(tmp_frontiers_);
}

void FrontierFinder::expandFrontier(const Eigen::Vector3i& first) {
  std::queue<Eigen::Vector3i> cell_queue;
  std::vector<Eigen::Vector3d> expanded;
  Eigen::Vector3d pos;

  edt_env_->sdf_map_->indexToPos(first, pos);
  expanded.push_back(pos);
  cell_queue.push(first);
  frontier_flag_[toadr(first)] = 1;

  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);
    for (auto nbr : nbrs) {
      int adr = toadr(nbr);
      if (frontier_flag_[adr] == 1 || !edt_env_->sdf_map_->isInBox(nbr) ||
          !(knownfree(nbr) && isNeighborUnknown(nbr)))
        continue;

      edt_env_->sdf_map_->indexToPos(nbr, pos);
      if (pos[2] < 0.4) continue;
      expanded.push_back(pos);
      cell_queue.push(nbr);
      frontier_flag_[adr] = 1;
    }
  }
  if ((int)expanded.size() > cluster_min_) {
    Frontier frontier;
    frontier.cells_ = expanded;
    computeFrontierInfo(frontier);
    tmp_frontiers_.push_back(frontier);
  }
}

void FrontierFinder::splitLargeFrontiers(std::list<Frontier>& frontiers) {
  std::list<Frontier> splits, tmps;
  for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
    if (splitHorizontally(*it, splits)) {
      tmps.insert(tmps.end(), splits.begin(), splits.end());
      splits.clear();
    } else
      tmps.push_back(*it);
  }
  frontiers = tmps;
}

bool FrontierFinder::splitHorizontally(const Frontier& frontier, std::list<Frontier>& splits) {
  auto mean = frontier.average_.head<2>();
  bool need_split = false;
  for (auto cell : frontier.filtered_cells_) {
    if ((cell.head<2>() - mean).norm() > cluster_size_xy_) {
      need_split = true;
      break;
    }
  }
  if (!need_split) return false;

  Eigen::Matrix2d cov;
  cov.setZero();
  for (auto cell : frontier.filtered_cells_) {
    Eigen::Vector2d diff = cell.head<2>() - mean;
    cov += diff * diff.transpose();
  }
  cov /= double(frontier.filtered_cells_.size());

  Eigen::EigenSolver<Eigen::Matrix2d> es(cov);
  auto values = es.eigenvalues().real();
  auto vectors = es.eigenvectors().real();
  int max_idx = 0;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  Eigen::Vector2d first_pc = vectors.col(max_idx);

  Frontier ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    if ((cell.head<2>() - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  std::list<Frontier> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  } else
    splits.push_back(ftr1);

  if (splitHorizontally(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

bool FrontierFinder::isFrontierChanged(const Frontier& ft) {
  for (auto cell : ft.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && isNeighborUnknown(idx))) return true;
  }
  return false;
}

void FrontierFinder::computeFrontierInfo(Frontier& ftr) {
  ftr.average_.setZero();
  ftr.box_max_ = ftr.cells_.front();
  ftr.box_min_ = ftr.cells_.front();
  for (auto cell : ftr.cells_) {
    ftr.average_ += cell;
    for (int i = 0; i < 3; ++i) {
      ftr.box_min_[i] = std::min(ftr.box_min_[i], cell[i]);
      ftr.box_max_[i] = std::max(ftr.box_max_[i], cell[i]);
    }
  }
  ftr.average_ /= double(ftr.cells_.size());
  downsample(ftr.cells_, ftr.filtered_cells_);
}

void FrontierFinder::computeFrontiersToVisit() {
  first_new_ftr_ = frontiers_.end();
  int new_num = 0;
  int new_dormant_num = 0;
  for (auto& tmp_ftr : tmp_frontiers_) {
    sampleViewpoints(tmp_ftr);
    if (!tmp_ftr.viewpoints_.empty()) {
      ++new_num;
      std::list<Frontier>::iterator inserted = frontiers_.insert(frontiers_.end(), tmp_ftr);
      std::sort(inserted->viewpoints_.begin(), inserted->viewpoints_.end(),
                [](const Viewpoint& v1, const Viewpoint& v2) { return v1.visib_num_ > v2.visib_num_; });
      if (first_new_ftr_ == frontiers_.end()) first_new_ftr_ = inserted;
    } else {
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
    }
  }
  int idx = 0;
  for (auto& ft : frontiers_) {
    ft.id_ = idx++;
  }
}

void FrontierFinder::getTopViewpointsInfo(const Eigen::Vector3d& cur_pos, std::vector<Eigen::Vector3d>& points,
                                           std::vector<double>& yaws, std::vector<Eigen::Vector3d>& averages) {
  points.clear();
  yaws.clear();
  averages.clear();
  for (auto frontier : frontiers_) {
    bool no_view = true;
    for (auto view : frontier.viewpoints_) {
      if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
      points.push_back(view.pos_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
      no_view = false;
      break;
    }
    if (no_view) {
      auto view = frontier.viewpoints_.front();
      points.push_back(view.pos_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
    }
  }
}

std::vector<FrontierInfo> FrontierFinder::getFrontierInfos(const Eigen::Vector3d& agent_pos) {
  std::vector<FrontierInfo> infos;
  for (auto& ftr : frontiers_) {
    FrontierInfo info;
    info.id = ftr.id_;
    info.cells = ftr.cells_;
    info.average = ftr.average_;
    info.box_min = ftr.box_min_;
    info.box_max = ftr.box_max_;
    info.frontier_size = (int)ftr.cells_.size();

    // Get best viewpoint
    if (!ftr.viewpoints_.empty()) {
      // Sort by visibility (should already be sorted)
      Viewpoint best = ftr.viewpoints_.front();
      // If too close, try to find one farther away
      for (auto& vp : ftr.viewpoints_) {
        if ((vp.pos_ - agent_pos).norm() >= min_candidate_dist_) {
          best = vp;
          break;
        }
      }
      info.best_viewpoint_pos = best.pos_;
      info.best_viewpoint_yaw = best.yaw_;
      info.best_viewpoint_visib_num = best.visib_num_;
    } else {
      info.best_viewpoint_pos = ftr.average_;
      info.best_viewpoint_yaw = 0.0;
      info.best_viewpoint_visib_num = 0;
    }
    infos.push_back(info);
  }
  return infos;
}

void FrontierFinder::getFrontiers(std::vector<std::vector<Eigen::Vector3d>>& clusters) {
  clusters.clear();
  for (auto frontier : frontiers_)
    clusters.push_back(frontier.cells_);
}

void FrontierFinder::getFrontierBoxes(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes) {
  boxes.clear();
  for (auto frontier : frontiers_) {
    Eigen::Vector3d center = (frontier.box_max_ + frontier.box_min_) * 0.5;
    Eigen::Vector3d scale = frontier.box_max_ - frontier.box_min_;
    boxes.push_back(std::make_pair(center, scale));
  }
}

void FrontierFinder::sampleViewpoints(Frontier& frontier) {
  for (double rc = candidate_rmin_, dr = (candidate_rmax_ - candidate_rmin_) / candidate_rnum_;
       rc <= candidate_rmax_ + 1e-3; rc += dr)
    for (double phi = -M_PI; phi < M_PI; phi += candidate_dphi_) {
      const Eigen::Vector3d sample_pos = frontier.average_ + rc * Eigen::Vector3d(cos(phi), sin(phi), 0);

      if (!edt_env_->sdf_map_->isInBox(sample_pos) ||
          edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1 || isNearUnknown(sample_pos))
        continue;

      auto& cells = frontier.filtered_cells_;
      Eigen::Vector3d ref_dir = (cells.front() - sample_pos).normalized();
      double avg_yaw = 0.0;
      for (size_t i = 1; i < cells.size(); ++i) {
        Eigen::Vector3d dir = (cells[i] - sample_pos).normalized();
        double yaw = acos(std::min(1.0, std::max(-1.0, dir.dot(ref_dir))));
        if (ref_dir.cross(dir)[2] < 0) yaw = -yaw;
        avg_yaw += yaw;
      }
      avg_yaw = avg_yaw / cells.size() + atan2(ref_dir[1], ref_dir[0]);
      wrapYaw(avg_yaw);
      int visib_num = countVisibleCells(sample_pos, avg_yaw, cells);
      if (visib_num > min_visib_num_) {
        Viewpoint vp = {sample_pos, avg_yaw, visib_num};
        frontier.viewpoints_.push_back(vp);
      }
    }
}

bool FrontierFinder::isFrontierCovered() {
  Eigen::Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max);

  auto checkChanges = [&](const std::list<Frontier>& frontiers) {
    for (auto ftr : frontiers) {
      if (!haveOverlap(ftr.box_min_, ftr.box_max_, update_min, update_max)) continue;
      const int change_thresh = min_view_finish_fraction_ * ftr.cells_.size();
      int change_num = 0;
      for (auto cell : ftr.cells_) {
        Eigen::Vector3i idx;
        edt_env_->sdf_map_->posToIndex(cell, idx);
        if (!(knownfree(idx) && isNeighborUnknown(idx)) && ++change_num >= change_thresh)
          return true;
      }
    }
    return false;
  };

  if (checkChanges(frontiers_) || checkChanges(dormant_frontiers_)) return true;
  return false;
}

bool FrontierFinder::isNearUnknown(const Eigen::Vector3d& pos) {
  const int vox_num = floor(min_candidate_clearance_ / resolution_);
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -1; z <= 1; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) == SDFMap::UNKNOWN) return true;
      }
  return false;
}

int FrontierFinder::countVisibleCells(const Eigen::Vector3d& pos, const double& yaw,
                                       const std::vector<Eigen::Vector3d>& cluster) {
  percep_utils_->setPose(pos, yaw);
  int visib_num = 0;
  Eigen::Vector3i idx;
  for (auto cell : cluster) {
    if (!percep_utils_->insideFOV(cell)) continue;

    raycaster_->input(cell, pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visib = false;
        break;
      }
    }
    if (visib) visib_num += 1;
  }
  return visib_num;
}

void FrontierFinder::downsample(const std::vector<Eigen::Vector3d>& cluster_in,
                                 std::vector<Eigen::Vector3d>& cluster_out) {
  // Simple voxel grid filter without PCL
  double leaf_size = resolution_ * down_sample_;
  double inv_leaf = 1.0 / leaf_size;
  cluster_out.clear();

  // Use a set of quantized indices to deduplicate
  struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
  };
  struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
      size_t seed = 0;
      seed ^= std::hash<int>()(k.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= std::hash<int>()(k.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= std::hash<int>()(k.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
    }
  };

  std::unordered_map<VoxelKey, Eigen::Vector3d, VoxelKeyHash> voxel_map;
  for (auto cell : cluster_in) {
    VoxelKey key;
    key.x = (int)floor(cell[0] * inv_leaf);
    key.y = (int)floor(cell[1] * inv_leaf);
    key.z = (int)floor(cell[2] * inv_leaf);
    voxel_map[key] = cell;  // Keep last point in each voxel
  }
  cluster_out.reserve(voxel_map.size());
  for (auto& kv : voxel_map) {
    cluster_out.push_back(kv.second);
  }
}

void FrontierFinder::wrapYaw(double& yaw) {
  while (yaw < -M_PI) yaw += 2 * M_PI;
  while (yaw > M_PI) yaw -= 2 * M_PI;
}

std::vector<Eigen::Vector3i> FrontierFinder::sixNeighbors(const Eigen::Vector3i& voxel) {
  std::vector<Eigen::Vector3i> neighbors(6);
  neighbors[0] = voxel - Eigen::Vector3i(1, 0, 0);
  neighbors[1] = voxel + Eigen::Vector3i(1, 0, 0);
  neighbors[2] = voxel - Eigen::Vector3i(0, 1, 0);
  neighbors[3] = voxel + Eigen::Vector3i(0, 1, 0);
  neighbors[4] = voxel - Eigen::Vector3i(0, 0, 1);
  neighbors[5] = voxel + Eigen::Vector3i(0, 0, 1);
  return neighbors;
}

std::vector<Eigen::Vector3i> FrontierFinder::allNeighbors(const Eigen::Vector3i& voxel) {
  std::vector<Eigen::Vector3i> neighbors(26);
  int count = 0;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0) continue;
        neighbors[count++] = voxel + Eigen::Vector3i(x, y, z);
      }
  return neighbors;
}

bool FrontierFinder::isNeighborUnknown(const Eigen::Vector3i& voxel) {
  auto nbrs = sixNeighbors(voxel);
  for (auto nbr : nbrs) {
    if (edt_env_->sdf_map_->getOccupancy(nbr) == SDFMap::UNKNOWN) return true;
  }
  return false;
}

inline int FrontierFinder::toadr(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->toAddress(idx);
}

inline bool FrontierFinder::knownfree(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::FREE;
}

inline bool FrontierFinder::inmap(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->isInMap(idx);
}

bool FrontierFinder::haveOverlap(const Eigen::Vector3d& min1, const Eigen::Vector3d& max1,
                                  const Eigen::Vector3d& min2, const Eigen::Vector3d& max2) {
  for (int i = 0; i < 3; ++i) {
    if (std::max(min1[i], min2[i]) > std::min(max1[i], max2[i]) + 1e-3) return false;
  }
  return true;
}

}  // namespace fuel_rl
