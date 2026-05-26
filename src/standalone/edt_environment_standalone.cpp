#include "edt_environment_standalone.h"

namespace fuel_rl {

void EDTEnvironment::init() {
}

void EDTEnvironment::setMap(SDFMap::Ptr& map) {
  this->sdf_map_ = map;
  resolution_inv_ = 1.0 / sdf_map_->getResolution();
}

void EDTEnvironment::evaluateEDTWithGrad(const Eigen::Vector3d& pos, double time, double& dist,
                                          Eigen::Vector3d& grad) {
  dist = sdf_map_->getDistWithGrad(pos, grad);
}

double EDTEnvironment::evaluateCoarseEDT(Eigen::Vector3d& pos, double time) {
  return sdf_map_->getDistance(pos);
}

}  // namespace fuel_rl
