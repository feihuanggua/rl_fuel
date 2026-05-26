#ifndef EDT_ENVIRONMENT_STANDALONE_H
#define EDT_ENVIRONMENT_STANDALONE_H

#include <Eigen/Eigen>
#include <memory>
#include "sdf_map_standalone.h"

namespace fuel_rl {

class EDTEnvironment {
public:
  using Ptr = std::shared_ptr<EDTEnvironment>;

  EDTEnvironment() {}
  ~EDTEnvironment() {}

  void init();
  void setMap(SDFMap::Ptr& map);

  void evaluateEDTWithGrad(const Eigen::Vector3d& pos, double time, double& dist, Eigen::Vector3d& grad);
  double evaluateCoarseEDT(Eigen::Vector3d& pos, double time);

  SDFMap::Ptr sdf_map_;

private:
  double resolution_inv_;
};

}  // namespace fuel_rl

#endif  // EDT_ENVIRONMENT_STANDALONE_H
