# DSL
Project page: https://sites.google.com/view/dsl-ram-lab/

## Monocular Direct Sparse Localization in a Prior 3D Surfel Map
#### Authors: Haoyang Ye, Huaiyang Huang, and Ming Liu from [RAM-LAB](https://ram-lab.com/).

## Paper and Video
Related publications:
```
@inproceedings{ye2020monocular,
  title={Monocular direct sparse localization in a prior 3d surfel map},
  author={Ye, Haoyang and Huang, Huaiyang and Liu, Ming},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={8892--8898},
  year={2020},
  organization={IEEE}
}
@inproceedings{ye20213d,
  title={3D Surfel Map-Aided Visual Relocalization with Learned Descriptors},
  author={Ye, Haoyang and Huang, Huaiyang and Hutter, Marco and Sandy, Timothy and Liu, Ming},
  booktitle={2021 International Conference on Robotics and Automation (ICRA)},
  pages={5574-5581},
  year={2021},
  organization={IEEE}
}
```
Video:
https://www.youtube.com/watch?v=LTihCBGcURo

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

## Note
This implementation of DSL takes [Ceres Solver](http://ceres-solver.org/) as backend, which is different from the the implementation of the original paper with DSO-backend. This leads to different performance, i.e., speed and accuracy, compared to the reported results.

## Credits
This work is inspired from several open-source projects, such as [DSO](https://github.com/JakobEngel/dso), [DSM](https://github.com/jzubizarreta/dsm), [Elastic-Fusion](https://github.com/mp3guy/ElasticFusion), [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), [DBoW2](https://github.com/dorian3d/DBoW2), [NetVlad](https://www.di.ens.fr/willow/research/netvlad/), [LIO-mapping](https://github.com/hyye/lio-mapping) and etc.

## Licence
The source code is released under [GPL-3.0](https://www.gnu.org/licenses/).
