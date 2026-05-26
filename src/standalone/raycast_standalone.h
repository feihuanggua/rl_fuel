#ifndef RAYCAST_STANDALONE_H_
#define RAYCAST_STANDALONE_H_

#include <Eigen/Eigen>
#include <vector>

namespace fuel_rl {

int signum(int x);
double mod(double value, double modulus);
double intbound(double s, double ds);

void Raycast(const Eigen::Vector3d& start, const Eigen::Vector3d& end, const Eigen::Vector3d& min,
             const Eigen::Vector3d& max, int& output_points_cnt, Eigen::Vector3d* output);

void Raycast(const Eigen::Vector3d& start, const Eigen::Vector3d& end, const Eigen::Vector3d& min,
             const Eigen::Vector3d& max, std::vector<Eigen::Vector3d>* output);

class RayCaster {
private:
  Eigen::Vector3d start_, end_, direction_;
  Eigen::Vector3d min_, max_;
  int x_, y_, z_, endX_, endY_, endZ_;
  double maxDist_;
  double dx_, dy_, dz_;
  int stepX_, stepY_, stepZ_;
  double tMaxX_, tMaxY_, tMaxZ_;
  double tDeltaX_, tDeltaY_, tDeltaZ_;
  double dist_;
  int step_num_;
  double resolution_;
  Eigen::Vector3d offset_;
  Eigen::Vector3d half_;

public:
  RayCaster() {}
  ~RayCaster() {}

  void setParams(const double& res, const Eigen::Vector3d& origin);
  bool input(const Eigen::Vector3d& start, const Eigen::Vector3d& end);
  bool nextId(Eigen::Vector3i& idx);
  bool nextPos(Eigen::Vector3d& pos);
};

}  // namespace fuel_rl

#endif  // RAYCAST_STANDALONE_H_
