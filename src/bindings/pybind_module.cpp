#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "fuel_env_core.h"

namespace py = pybind11;

PYBIND11_MODULE(fuel_rl_core, m) {
  m.doc() = "FUEL RL Environment C++ Core";

  // Parameter structs
  py::class_<fuel_rl::SDFMapParams>(m, "SDFMapParams")
      .def(py::init<>())
      .def_readwrite("resolution", &fuel_rl::SDFMapParams::resolution)
      .def_readwrite("map_size_x", &fuel_rl::SDFMapParams::map_size_x)
      .def_readwrite("map_size_y", &fuel_rl::SDFMapParams::map_size_y)
      .def_readwrite("map_size_z", &fuel_rl::SDFMapParams::map_size_z)
      .def_readwrite("obstacles_inflation", &fuel_rl::SDFMapParams::obstacles_inflation)
      .def_readwrite("ground_height", &fuel_rl::SDFMapParams::ground_height)
      .def_readwrite("default_dist", &fuel_rl::SDFMapParams::default_dist)
      .def_readwrite("p_hit", &fuel_rl::SDFMapParams::p_hit)
      .def_readwrite("p_miss", &fuel_rl::SDFMapParams::p_miss)
      .def_readwrite("max_ray_length", &fuel_rl::SDFMapParams::max_ray_length)
      .def_readwrite("optimistic", &fuel_rl::SDFMapParams::optimistic)
      .def_readwrite("box_min_x", &fuel_rl::SDFMapParams::box_min_x)
      .def_readwrite("box_min_y", &fuel_rl::SDFMapParams::box_min_y)
      .def_readwrite("box_min_z", &fuel_rl::SDFMapParams::box_min_z)
      .def_readwrite("box_max_x", &fuel_rl::SDFMapParams::box_max_x)
      .def_readwrite("box_max_y", &fuel_rl::SDFMapParams::box_max_y)
      .def_readwrite("box_max_z", &fuel_rl::SDFMapParams::box_max_z);

  py::class_<fuel_rl::FrontierParams>(m, "FrontierParams")
      .def(py::init<>())
      .def_readwrite("cluster_min", &fuel_rl::FrontierParams::cluster_min)
      .def_readwrite("cluster_size_xy", &fuel_rl::FrontierParams::cluster_size_xy)
      .def_readwrite("candidate_rmin", &fuel_rl::FrontierParams::candidate_rmin)
      .def_readwrite("candidate_rmax", &fuel_rl::FrontierParams::candidate_rmax)
      .def_readwrite("min_visib_num", &fuel_rl::FrontierParams::min_visib_num);

  py::class_<fuel_rl::PerceptionParams>(m, "PerceptionParams")
      .def(py::init<>())
      .def_readwrite("top_angle", &fuel_rl::PerceptionParams::top_angle)
      .def_readwrite("left_angle", &fuel_rl::PerceptionParams::left_angle)
      .def_readwrite("right_angle", &fuel_rl::PerceptionParams::right_angle)
      .def_readwrite("max_dist", &fuel_rl::PerceptionParams::max_dist)
      .def_readwrite("skip_pixel", &fuel_rl::PerceptionParams::skip_pixel);

  py::class_<fuel_rl::AstarParams>(m, "AstarParams")
      .def(py::init<>())
      .def_readwrite("lambda_heu", &fuel_rl::AstarParams::lambda_heu)
      .def_readwrite("resolution_astar", &fuel_rl::AstarParams::resolution_astar)
      .def_readwrite("max_search_time", &fuel_rl::AstarParams::max_search_time)
      .def_readwrite("allocate_num", &fuel_rl::AstarParams::allocate_num);

  // FrontierInfo
  py::class_<fuel_rl::FrontierInfo>(m, "FrontierInfo")
      .def_readonly("id", &fuel_rl::FrontierInfo::id)
      .def_readonly("cells", &fuel_rl::FrontierInfo::cells)
      .def_readonly("average", &fuel_rl::FrontierInfo::average)
      .def_readonly("box_min", &fuel_rl::FrontierInfo::box_min)
      .def_readonly("box_max", &fuel_rl::FrontierInfo::box_max)
      .def_readonly("best_viewpoint_pos", &fuel_rl::FrontierInfo::best_viewpoint_pos)
      .def_readonly("best_viewpoint_yaw", &fuel_rl::FrontierInfo::best_viewpoint_yaw)
      .def_readonly("best_viewpoint_visib_num", &fuel_rl::FrontierInfo::best_viewpoint_visib_num)
      .def_readonly("frontier_size", &fuel_rl::FrontierInfo::frontier_size);

  // FuelEnvCore
  py::class_<fuel_rl::FuelEnvCore, fuel_rl::FuelEnvCore::Ptr>(m, "FuelEnvCore")
      .def(py::init<>())
      .def("init", &fuel_rl::FuelEnvCore::init)
      .def("load_map_from_pcd", &fuel_rl::FuelEnvCore::loadMapFromPCD)
      .def("load_map_from_points", &fuel_rl::FuelEnvCore::loadMapFromPoints)
      .def("reset_map", &fuel_rl::FuelEnvCore::resetMap)
      .def("simulate_observation", &fuel_rl::FuelEnvCore::simulateObservation)
      .def("input_hit_points", &fuel_rl::FuelEnvCore::inputHitPoints)
      .def("detect_frontiers", &fuel_rl::FuelEnvCore::detectFrontiers)
      .def("plan_path", &fuel_rl::FuelEnvCore::planPath)
      .def("compute_path_cost", &fuel_rl::FuelEnvCore::computePathCost)
      .def("count_visible_cells", &fuel_rl::FuelEnvCore::countVisibleCells)
      .def("get_occupancy", &fuel_rl::FuelEnvCore::getOccupancy)
      .def("get_distance", &fuel_rl::FuelEnvCore::getDistance)
      .def("get_resolution", &fuel_rl::FuelEnvCore::getResolution)
      .def("get_region", [](fuel_rl::FuelEnvCore& c) {
          Eigen::Vector3d ori, sz;
          c.getRegion(ori, sz);
          return std::make_tuple(ori, sz);
      })
      .def("get_box", [](fuel_rl::FuelEnvCore& c) {
          Eigen::Vector3d bmin, bmax;
          c.getBox(bmin, bmax);
          return std::make_tuple(bmin, bmax);
      })
      .def("count_unknown_voxels", &fuel_rl::FuelEnvCore::countUnknownVoxels)
      .def("count_free_voxels", &fuel_rl::FuelEnvCore::countFreeVoxels)
      .def("count_occupied_voxels", &fuel_rl::FuelEnvCore::countOccupiedVoxels)
      .def("get_exploration_progress", &fuel_rl::FuelEnvCore::getExplorationProgress)
      .def("get_local_voxel_grid", &fuel_rl::FuelEnvCore::getLocalVoxelGrid)
      .def("get_occupancy_slice_2d", &fuel_rl::FuelEnvCore::getOccupancySlice2D)
      .def("get_map_voxel_num", &fuel_rl::FuelEnvCore::getMapVoxelNum)
      .def("set_ground_plane", &fuel_rl::FuelEnvCore::setGroundPlane,
           py::arg("enable"), py::arg("height") = 0.0);
}
