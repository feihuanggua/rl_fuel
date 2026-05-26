#include "sdf_map_standalone.h"

namespace fuel_rl {

SDFMap::SDFMap() {}

SDFMap::~SDFMap() {}

void SDFMap::init(const SDFMapParams& params) {
  mp_.reset(new MapParam);
  md_.reset(new MapData);

  mp_->resolution_ = params.resolution;
  mp_->local_bound_inflate_ = std::max(mp_->resolution_, params.local_bound_inflate);
  mp_->resolution_inv_ = 1.0 / mp_->resolution_;
  mp_->ground_height_ = params.ground_height;
  mp_->map_origin_ = Eigen::Vector3d(-params.map_size_x / 2.0, -params.map_size_y / 2.0, mp_->ground_height_);
  mp_->map_size_ = Eigen::Vector3d(params.map_size_x, params.map_size_y, params.map_size_z);
  for (int i = 0; i < 3; ++i)
    mp_->map_voxel_num_(i) = ceil(mp_->map_size_(i) / mp_->resolution_);
  mp_->map_min_boundary_ = mp_->map_origin_;
  mp_->map_max_boundary_ = mp_->map_origin_ + mp_->map_size_;

  mp_->obstacles_inflation_ = params.obstacles_inflation;
  mp_->default_dist_ = params.default_dist;
  mp_->optimistic_ = params.optimistic;
  mp_->signed_dist_ = params.signed_dist;

  mp_->p_hit_ = params.p_hit;
  mp_->p_miss_ = params.p_miss;
  mp_->p_min_ = params.p_min;
  mp_->p_max_ = params.p_max;
  mp_->p_occ_ = params.p_occ;
  mp_->max_ray_length_ = params.max_ray_length;
  mp_->virtual_ceil_height_ = params.virtual_ceil_height;
  mp_->local_map_margin_ = params.local_map_margin;

  auto logit = [](const double& x) { return log(x / (1 - x)); };
  mp_->prob_hit_log_ = logit(mp_->p_hit_);
  mp_->prob_miss_log_ = logit(mp_->p_miss_);
  mp_->clamp_min_log_ = logit(mp_->p_min_);
  mp_->clamp_max_log_ = logit(mp_->p_max_);
  mp_->min_occupancy_log_ = logit(mp_->p_occ_);
  mp_->unknown_flag_ = 0.01;

  int buffer_size = mp_->map_voxel_num_(0) * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2);
  md_->occupancy_buffer_ = std::vector<double>(buffer_size, mp_->clamp_min_log_ - mp_->unknown_flag_);
  md_->occupancy_buffer_inflate_ = std::vector<char>(buffer_size, 0);
  md_->distance_buffer_neg_ = std::vector<double>(buffer_size, mp_->default_dist_);
  md_->distance_buffer_ = std::vector<double>(buffer_size, mp_->default_dist_);
  md_->count_hit_and_miss_ = std::vector<short>(buffer_size, 0);
  md_->count_hit_ = std::vector<short>(buffer_size, 0);
  md_->count_miss_ = std::vector<short>(buffer_size, 0);
  md_->flag_rayend_ = std::vector<char>(buffer_size, -1);
  md_->flag_visited_ = std::vector<char>(buffer_size, -1);
  md_->tmp_buffer1_ = std::vector<double>(buffer_size, 0);
  md_->tmp_buffer2_ = std::vector<double>(buffer_size, 0);
  md_->raycast_num_ = 0;
  md_->reset_updated_box_ = true;
  md_->update_min_ = md_->update_max_ = Eigen::Vector3d(0, 0, 0);
  md_->local_updated_ = false;

  // Bounding box
  if (params.box_min_x != 0 || params.box_min_y != 0 || params.box_min_z != 0 ||
      params.box_max_x != 0 || params.box_max_y != 0 || params.box_max_z != 0) {
    mp_->box_mind_ = Eigen::Vector3d(params.box_min_x, params.box_min_y, params.box_min_z);
    mp_->box_maxd_ = Eigen::Vector3d(params.box_max_x, params.box_max_y, params.box_max_z);
  } else {
    mp_->box_mind_ = mp_->map_min_boundary_;
    mp_->box_maxd_ = mp_->map_max_boundary_;
  }
  posToIndex(mp_->box_mind_, mp_->box_min_);
  posToIndex(mp_->box_maxd_, mp_->box_max_);

  caster_.reset(new RayCaster);
  caster_->setParams(mp_->resolution_, mp_->map_origin_);
}

void SDFMap::resetBuffer() {
  resetBuffer(mp_->map_min_boundary_, mp_->map_max_boundary_);
  md_->local_bound_min_ = Eigen::Vector3i::Zero();
  md_->local_bound_max_ = mp_->map_voxel_num_ - Eigen::Vector3i::Ones();
}

void SDFMap::resetBuffer(const Eigen::Vector3d& min_pos, const Eigen::Vector3d& max_pos) {
  Eigen::Vector3i min_id, max_id;
  posToIndex(min_pos, min_id);
  posToIndex(max_pos, max_id);
  boundIndex(min_id);
  boundIndex(max_id);

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        md_->occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
        md_->distance_buffer_[toAddress(x, y, z)] = mp_->default_dist_;
      }
}

template <typename F_get_val, typename F_set_val>
void SDFMap::fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim) {
  int v[mp_->map_voxel_num_(dim)];
  double z[mp_->map_voxel_num_(dim) + 1];

  int k = start;
  v[start] = start;
  z[start] = -std::numeric_limits<double>::max();
  z[start + 1] = std::numeric_limits<double>::max();

  for (int q = start + 1; q <= end; q++) {
    k++;
    double s;

    do {
      k--;
      s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
    } while (s <= z[k]);

    k++;

    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<double>::max();
  }

  k = start;

  for (int q = start; q <= end; q++) {
    while (z[k + 1] < q)
      k++;
    double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
    f_set_val(q, val);
  }
}

void SDFMap::updateESDF3d() {
  Eigen::Vector3i min_esdf = md_->local_bound_min_;
  Eigen::Vector3i max_esdf = md_->local_bound_max_;

  if (mp_->optimistic_) {
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
      for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
        fillESDF(
            [&](int z) {
              return md_->occupancy_buffer_inflate_[toAddress(x, y, z)] == 1 ?
                  0 :
                  std::numeric_limits<double>::max();
            },
            [&](int z, double val) { md_->tmp_buffer1_[toAddress(x, y, z)] = val; }, min_esdf[2],
            max_esdf[2], 2);
      }
  } else {
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
      for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
        fillESDF(
            [&](int z) {
              int adr = toAddress(x, y, z);
              return (md_->occupancy_buffer_inflate_[adr] == 1 ||
                      md_->occupancy_buffer_[adr] < mp_->clamp_min_log_ - 1e-3) ?
                  0 :
                  std::numeric_limits<double>::max();
            },
            [&](int z, double val) { md_->tmp_buffer1_[toAddress(x, y, z)] = val; }, min_esdf[2],
            max_esdf[2], 2);
      }
  }

  for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF(
          [&](int y) { return md_->tmp_buffer1_[toAddress(x, y, z)]; },
          [&](int y, double val) { md_->tmp_buffer2_[toAddress(x, y, z)] = val; }, min_esdf[1],
          max_esdf[1], 1);
    }
  for (int y = min_esdf[1]; y <= max_esdf[1]; y++)
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF(
          [&](int x) { return md_->tmp_buffer2_[toAddress(x, y, z)]; },
          [&](int x, double val) {
            md_->distance_buffer_[toAddress(x, y, z)] = mp_->resolution_ * std::sqrt(val);
          },
          min_esdf[0], max_esdf[0], 0);
    }

  if (mp_->signed_dist_) {
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
      for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
        fillESDF(
            [&](int z) {
              return md_->occupancy_buffer_inflate_
                          [x * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2) +
                           y * mp_->map_voxel_num_(2) + z] == 0 ?
                  0 :
                  std::numeric_limits<double>::max();
            },
            [&](int z, double val) { md_->tmp_buffer1_[toAddress(x, y, z)] = val; }, min_esdf[2],
            max_esdf[2], 2);
      }
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
      for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
        fillESDF(
            [&](int y) { return md_->tmp_buffer1_[toAddress(x, y, z)]; },
            [&](int y, double val) { md_->tmp_buffer2_[toAddress(x, y, z)] = val; }, min_esdf[1],
            max_esdf[1], 1);
      }
    for (int y = min_esdf[1]; y <= max_esdf[1]; y++)
      for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
        fillESDF(
            [&](int x) { return md_->tmp_buffer2_[toAddress(x, y, z)]; },
            [&](int x, double val) {
              md_->distance_buffer_neg_[toAddress(x, y, z)] = mp_->resolution_ * std::sqrt(val);
            },
            min_esdf[0], max_esdf[0], 0);
      }
    for (int x = min_esdf(0); x <= max_esdf(0); ++x)
      for (int y = min_esdf(1); y <= max_esdf(1); ++y)
        for (int z = min_esdf(2); z <= max_esdf(2); ++z) {
          int idx = toAddress(x, y, z);
          if (md_->distance_buffer_neg_[idx] > 0.0)
            md_->distance_buffer_[idx] += (-md_->distance_buffer_neg_[idx] + mp_->resolution_);
        }
  }
}

void SDFMap::setCacheOccupancy(const int& adr, const int& occ) {
  if (md_->count_hit_[adr] == 0 && md_->count_miss_[adr] == 0) md_->cache_voxel_.push(adr);

  if (occ == 0)
    md_->count_miss_[adr] = 1;
  else if (occ == 1)
    md_->count_hit_[adr] += 1;
}

void SDFMap::inputPointCloud(const std::vector<Eigen::Vector3d>& points, const Eigen::Vector3d& camera_pos) {
  int point_num = points.size();
  if (point_num == 0) return;
  md_->raycast_num_ += 1;

  Eigen::Vector3d update_min = camera_pos;
  Eigen::Vector3d update_max = camera_pos;
  if (md_->reset_updated_box_) {
    md_->update_min_ = camera_pos;
    md_->update_max_ = camera_pos;
    md_->reset_updated_box_ = false;
  }

  Eigen::Vector3d pt_w;
  Eigen::Vector3i idx;
  int vox_adr;
  double length;
  for (int i = 0; i < point_num; ++i) {
    pt_w = points[i];
    int tmp_flag;
    if (!isInMap(pt_w)) {
      pt_w = closetPointInMap(pt_w, camera_pos);
      length = (pt_w - camera_pos).norm();
      if (length > mp_->max_ray_length_)
        pt_w = (pt_w - camera_pos) / length * mp_->max_ray_length_ + camera_pos;
      if (pt_w[2] < 0.2) continue;
      tmp_flag = 0;
    } else {
      length = (pt_w - camera_pos).norm();
      if (length > mp_->max_ray_length_) {
        pt_w = (pt_w - camera_pos) / length * mp_->max_ray_length_ + camera_pos;
        if (pt_w[2] < 0.2) continue;
        tmp_flag = 0;
      } else
        tmp_flag = 1;
    }
    posToIndex(pt_w, idx);
    vox_adr = toAddress(idx);
    setCacheOccupancy(vox_adr, tmp_flag);

    for (int k = 0; k < 3; ++k) {
      update_min[k] = std::min(update_min[k], pt_w[k]);
      update_max[k] = std::max(update_max[k], pt_w[k]);
    }

    if (md_->flag_rayend_[vox_adr] == md_->raycast_num_)
      continue;
    else
      md_->flag_rayend_[vox_adr] = md_->raycast_num_;

    caster_->input(pt_w, camera_pos);
    caster_->nextId(idx);
    while (caster_->nextId(idx))
      setCacheOccupancy(toAddress(idx), 0);
  }

  Eigen::Vector3d bound_inf(mp_->local_bound_inflate_, mp_->local_bound_inflate_, 0);
  posToIndex(update_max + bound_inf, md_->local_bound_max_);
  posToIndex(update_min - bound_inf, md_->local_bound_min_);
  boundIndex(md_->local_bound_min_);
  boundIndex(md_->local_bound_max_);
  md_->local_updated_ = true;

  for (int k = 0; k < 3; ++k) {
    md_->update_min_[k] = std::min(update_min[k], md_->update_min_[k]);
    md_->update_max_[k] = std::max(update_max[k], md_->update_max_[k]);
  }

  while (!md_->cache_voxel_.empty()) {
    int adr = md_->cache_voxel_.front();
    md_->cache_voxel_.pop();
    double log_odds_update =
        md_->count_hit_[adr] >= md_->count_miss_[adr] ? mp_->prob_hit_log_ : mp_->prob_miss_log_;
    md_->count_hit_[adr] = md_->count_miss_[adr] = 0;
    if (md_->occupancy_buffer_[adr] < mp_->clamp_min_log_ - 1e-3)
      md_->occupancy_buffer_[adr] = mp_->min_occupancy_log_;

    md_->occupancy_buffer_[adr] = std::min(
        std::max(md_->occupancy_buffer_[adr] + log_odds_update, mp_->clamp_min_log_),
        mp_->clamp_max_log_);
  }
}

void SDFMap::injectObstacles(const std::vector<Eigen::Vector3d>& points) {
  for (const auto& pt : points) {
    if (!isInMap(pt)) continue;
    Eigen::Vector3i id;
    posToIndex(pt, id);
    int adr = toAddress(id);
    // Directly mark as occupied in the occupancy buffer
    md_->occupancy_buffer_[adr] = mp_->clamp_max_log_;
  }
  // Update local bounds to cover the whole map for inflation
  md_->local_bound_min_ = Eigen::Vector3i::Zero();
  md_->local_bound_max_ = mp_->map_voxel_num_ - Eigen::Vector3i::Ones();
  clearAndInflateLocalMap();
  updateESDF3d();
}

void SDFMap::addGroundPlane(double ground_z) {
  // Fill all voxels from map bottom up to ground_z as occupied
  // This ensures nothing below the ground level can be unknown/free
  Eigen::Vector3i box_min, box_max;
  posToIndex(mp_->box_mind_, box_min);
  posToIndex(mp_->box_maxd_, box_max);
  boundIndex(box_min);
  boundIndex(box_max);

  Eigen::Vector3i ground_idx;
  posToIndex(Eigen::Vector3d(0, 0, ground_z), ground_idx);
  int max_z = std::min(ground_idx[2], box_max[2]);
  int min_z = 0;  // Start from map bottom

  for (int x = box_min[0]; x <= box_max[0]; ++x)
    for (int y = box_min[1]; y <= box_max[1]; ++y)
      for (int z = min_z; z <= max_z; ++z) {
        md_->occupancy_buffer_[toAddress(x, y, z)] = mp_->clamp_max_log_;
      }

  // Update local bounds to cover the full box for inflation
  md_->local_bound_min_ = Eigen::Vector3i::Zero();
  md_->local_bound_max_ = mp_->map_voxel_num_ - Eigen::Vector3i::Ones();
  clearAndInflateLocalMap();
  updateESDF3d();
}

Eigen::Vector3d SDFMap::closetPointInMap(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt) {
  Eigen::Vector3d diff = pt - camera_pt;
  Eigen::Vector3d max_tc = mp_->map_max_boundary_ - camera_pt;
  Eigen::Vector3d min_tc = mp_->map_min_boundary_ - camera_pt;
  double min_t = 1000000;
  for (int i = 0; i < 3; ++i) {
    if (fabs(diff[i]) > 0) {
      double t1 = max_tc[i] / diff[i];
      if (t1 > 0 && t1 < min_t) min_t = t1;
      double t2 = min_tc[i] / diff[i];
      if (t2 > 0 && t2 < min_t) min_t = t2;
    }
  }
  return camera_pt + (min_t - 1e-3) * diff;
}

void SDFMap::clearAndInflateLocalMap() {
  int inf_step = ceil(mp_->obstacles_inflation_ / mp_->resolution_);
  std::vector<Eigen::Vector3i> inf_pts(pow(2 * inf_step + 1, 3));

  for (int x = md_->local_bound_min_(0); x <= md_->local_bound_max_(0); ++x)
    for (int y = md_->local_bound_min_(1); y <= md_->local_bound_max_(1); ++y)
      for (int z = md_->local_bound_min_(2); z <= md_->local_bound_max_(2); ++z) {
        md_->occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
      }

  for (int x = md_->local_bound_min_(0); x <= md_->local_bound_max_(0); ++x)
    for (int y = md_->local_bound_min_(1); y <= md_->local_bound_max_(1); ++y)
      for (int z = md_->local_bound_min_(2); z <= md_->local_bound_max_(2); ++z) {
        int id1 = toAddress(x, y, z);
        if (md_->occupancy_buffer_[id1] > mp_->min_occupancy_log_) {
          inflatePoint(Eigen::Vector3i(x, y, z), inf_step, inf_pts);
          for (auto inf_pt : inf_pts) {
            int idx_inf = toAddress(inf_pt);
            if (idx_inf >= 0 &&
                idx_inf < mp_->map_voxel_num_(0) * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2)) {
              md_->occupancy_buffer_inflate_[idx_inf] = 1;
            }
          }
        }
      }

  if (mp_->virtual_ceil_height_ > -0.5) {
    int ceil_id = floor((mp_->virtual_ceil_height_ - mp_->map_origin_(2)) * mp_->resolution_inv_);
    for (int x = md_->local_bound_min_(0); x <= md_->local_bound_max_(0); ++x)
      for (int y = md_->local_bound_min_(1); y <= md_->local_bound_max_(1); ++y) {
        md_->occupancy_buffer_[toAddress(x, y, ceil_id)] = mp_->clamp_max_log_;
      }
  }
}

double SDFMap::getResolution() { return mp_->resolution_; }

int SDFMap::getVoxelNum() { return mp_->map_voxel_num_[0] * mp_->map_voxel_num_[1] * mp_->map_voxel_num_[2]; }

void SDFMap::getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size) { ori = mp_->map_origin_, size = mp_->map_size_; }

void SDFMap::getBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax) { bmin = mp_->box_mind_; bmax = mp_->box_maxd_; }

void SDFMap::getUpdatedBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax, bool reset) {
  bmin = md_->update_min_;
  bmax = md_->update_max_;
  if (reset) md_->reset_updated_box_ = true;
}

double SDFMap::getDistWithGrad(const Eigen::Vector3d& pos, Eigen::Vector3d& grad) {
  if (!isInMap(pos)) {
    grad.setZero();
    return 0;
  }

  Eigen::Vector3d pos_m = pos - 0.5 * mp_->resolution_ * Eigen::Vector3d::Ones();
  Eigen::Vector3i idx;
  posToIndex(pos_m, idx);
  Eigen::Vector3d idx_pos, diff;
  indexToPos(idx, idx_pos);
  diff = (pos - idx_pos) * mp_->resolution_inv_;

  double values[2][2][2];
  for (int x = 0; x < 2; x++)
    for (int y = 0; y < 2; y++)
      for (int z = 0; z < 2; z++) {
        Eigen::Vector3i current_idx = idx + Eigen::Vector3i(x, y, z);
        values[x][y][z] = getDistance(current_idx);
      }

  double v00 = (1 - diff[0]) * values[0][0][0] + diff[0] * values[1][0][0];
  double v01 = (1 - diff[0]) * values[0][0][1] + diff[0] * values[1][0][1];
  double v10 = (1 - diff[0]) * values[0][1][0] + diff[0] * values[1][1][0];
  double v11 = (1 - diff[0]) * values[0][1][1] + diff[0] * values[1][1][1];
  double v0 = (1 - diff[1]) * v00 + diff[1] * v10;
  double v1 = (1 - diff[1]) * v01 + diff[1] * v11;
  double dist = (1 - diff[2]) * v0 + diff[2] * v1;

  grad[2] = (v1 - v0) * mp_->resolution_inv_;
  grad[1] = ((1 - diff[2]) * (v10 - v00) + diff[2]) * (v11 - v01) * mp_->resolution_inv_;
  grad[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1][0][0] - values[0][0][0]);
  grad[0] += (1 - diff[2]) * diff[1] * (values[1][1][0] - values[0][1][0]);
  grad[0] += diff[2] * (1 - diff[1]) * (values[1][0][1] - values[0][0][1]);
  grad[0] += diff[2] * diff[1] * (values[1][1][1] - values[0][1][1]);
  grad[0] *= mp_->resolution_inv_;

  return dist;
}

// RL-specific methods
std::vector<int8_t> SDFMap::getLocalVoxelGrid(const Eigen::Vector3d& center, int grid_size) {
  int half = grid_size / 2;
  std::vector<int8_t> grid(grid_size * grid_size * grid_size, -1);

  Eigen::Vector3i center_idx;
  posToIndex(center, center_idx);

  for (int dx = -half; dx <= half; ++dx)
    for (int dy = -half; dy <= half; ++dy)
      for (int dz = -half; dz <= half; ++dz) {
        Eigen::Vector3i query_idx = center_idx + Eigen::Vector3i(dx, dy, dz);
        int gx = dx + half, gy = dy + half, gz = dz + half;
        int gidx = gx * grid_size * grid_size + gy * grid_size + gz;

        if (!isInMap(query_idx)) {
          grid[gidx] = -1;  // out of map
          continue;
        }
        int occ = getOccupancy(query_idx);
        int inf = getInflateOccupancy(query_idx);
        if (inf == 1)
          grid[gidx] = 3;  // inflated occupied
        else if (occ == OCCUPIED)
          grid[gidx] = 2;  // occupied
        else if (occ == FREE)
          grid[gidx] = 1;  // free
        else
          grid[gidx] = 0;  // unknown
      }

  return grid;
}

std::vector<int8_t> SDFMap::getOccupancySlice2D(double height) {
  Eigen::Vector3i h_idx;
  posToIndex(Eigen::Vector3d(0, 0, height), h_idx);
  int z = h_idx[2];
  if (z < 0 || z >= mp_->map_voxel_num_[2]) return {};

  int nx = mp_->map_voxel_num_[0];
  int ny = mp_->map_voxel_num_[1];
  std::vector<int8_t> slice(nx * ny);

  for (int x = 0; x < nx; ++x)
    for (int y = 0; y < ny; ++y) {
      int occ = getOccupancy(Eigen::Vector3i(x, y, z));
      int inf = getInflateOccupancy(Eigen::Vector3i(x, y, z));
      if (inf == 1)
        slice[x * ny + y] = 3;
      else if (occ == OCCUPIED)
        slice[x * ny + y] = 2;
      else if (occ == FREE)
        slice[x * ny + y] = 1;
      else
        slice[x * ny + y] = 0;
    }

  return slice;
}

int SDFMap::countUnknownVoxels() const {
  int count = 0;
  for (size_t i = 0; i < md_->occupancy_buffer_.size(); ++i) {
    if (md_->occupancy_buffer_[i] < mp_->clamp_min_log_ - 1e-3) count++;
  }
  return count;
}

int SDFMap::countFreeVoxels() const {
  int count = 0;
  for (size_t i = 0; i < md_->occupancy_buffer_.size(); ++i) {
    double occ = md_->occupancy_buffer_[i];
    if (occ >= mp_->clamp_min_log_ - 1e-3 && occ <= mp_->min_occupancy_log_) count++;
  }
  return count;
}

int SDFMap::countOccupiedVoxels() const {
  int count = 0;
  for (size_t i = 0; i < md_->occupancy_buffer_.size(); ++i) {
    if (md_->occupancy_buffer_[i] > mp_->min_occupancy_log_) count++;
  }
  return count;
}

double SDFMap::getExplorationProgress() const {
  int total = md_->occupancy_buffer_.size();
  if (total == 0) return 0.0;
  int known = total - countUnknownVoxels();
  return static_cast<double>(known) / total;
}

}  // namespace fuel_rl
