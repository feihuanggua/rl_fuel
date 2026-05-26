from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

ext_modules = [
    Pybind11Extension(
        "fuel_rl_core",
        sources=[
            "src/standalone/raycast_standalone.cpp",
            "src/standalone/sdf_map_standalone.cpp",
            "src/standalone/edt_environment_standalone.cpp",
            "src/standalone/perception_utils_standalone.cpp",
            "src/standalone/astar_standalone.cpp",
            "src/standalone/frontier_finder_standalone.cpp",
            "src/standalone/fuel_env_core.cpp",
            "src/bindings/pybind_module.cpp",
        ],
        include_dirs=[
            "src/standalone",
            "/usr/include/eigen3",
            "/usr/include/pcl-1.10",
        ],
        extra_compile_args=["-std=c++14", "-O3", "-fPIC"],
        libraries=["pcl_common", "pcl_io", "pcl_kdtree", "pcl_search"],
        library_dirs=["/usr/lib/x86_64-linux-gnu"],
    ),
]

setup(
    name="fuel-rl",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.24",
        "matplotlib>=3.5",
    ],
    extras_require={
        "train": ["stable-baselines3>=2.0"],
        "vis3d": ["open3d"],
    },
)
