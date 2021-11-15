# DSL

## Monocular Direct Sparse Localization in a Prior 3D Surfel Map
#### Authors: Haoyang Ye, Huaiyang Huang, and Ming Liu from [RAM-LAB](https://ram-lab.com/).

## Dependency
1. [Pangolin](https://github.com/stevenlovegrove/Pangolin).
2. [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal).
3. [Ceres-solver](http://ceres-solver.org/installation.html#linux).
4. [PCL](http://www.pointclouds.org/downloads/), the default version accompanying by ROS.
5. [OpenCV](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html), the default version accompanying by ROS.

## Build
1. `mkdir build && cd build`
2. `cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo` 
3. `make -j8`

## Example
The sample config file can be downloaded from [this link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeab_connect_ust_hk/EXDlfYJgCclHuxvTiEcfpbIBJmNAwdE1soquwUeHGuaItw?e=63oiEP).

To run the example:
```shell
[path_to_build]/src/dsl_main --path "[path_to_dataset]/left_pinhole"
```

## Preparing Your Own Data
1. Collect LiDAR and camera data.
2. Build LiDAR map and obtain LiDAR poses (the poses are not necessary).
3. Pre-process LiDAR map to make the `[path_to_dataset]/*.pcd` map file contains `normal_x, normal_y, normal_z` fields (downsample & normal estimation).
4. Extract and undistort images into `[path_to_dataset]/images`.
5. Set the first camera pose to `initial_pose` and other camera parameters in `[path_to_dataset]/config.yaml`.

## Todo:
- [ ] Add neighbor covisible keyframes
- [ ] SuperPoint/SIFT matcher?