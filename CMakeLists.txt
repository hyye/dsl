cmake_minimum_required(VERSION 3.7.2)

project(dsl)

set(CMAKE_CXX_STANDARD 17)

add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS
#"${SSE_FLAGS} -g -march=native"
##   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
#)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Boost REQUIRED COMPONENTS system filesystem thread)
include_directories(${Boost_INCLUDE_DIRS})

list(APPEND CMAKE_PREFIX_PATH "/mnt/HDD/Softwares/opencv-3.4.11/install/share/OpenCV")
find_package(OpenCV 3.4.11 REQUIRED CONFIG)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS system)
include_directories(${Boost_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/3rdparty/libtorch/")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
message(${TORCH_INCLUDE_DIRS})

set(3rdparty_INCLUDE_DIRS
3rdparty/googletest/googletest/include
3rdparty/googletest/googlemock/include
3rdparty/Sophus
3rdparty/nanoflann/include
3rdparty/DBoW2/include
3rdparty/fmt/include
)

include_directories(${3rdparty_INCLUDE_DIRS})
include_directories(include)
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(3rdparty/googletest)
set(Google_LIBRARIES
${GFLAGS_LIBRARIES}
${GLOG_LIBRARIES}
gmock
gtest gtest_main
)

add_subdirectory(3rdparty/DBoW2)
add_subdirectory(3rdparty/fmt)

set(ThreeParty_LIBS
DBoW2
fmt
)

set(Ext_LIBRARIES
${ThreeParty_LIBS}
${CERES_LIBRARIES}
${GFLAGS_LIBRARIES}
${OpenCV_LIBRARIES}
${PCL_LIBRARIES}
${Pangolin_LIBRARIES}
${CUDA_LIBRARIES}
${TORCH_LIBRARIES}
)

enable_testing()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC;")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

set(SHADER_DIR "${PROJECT_SOURCE_DIR}/include/map_render/shaders" CACHE PATH "Where the shaders live")
set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-DSHADER_DIR=${SHADER_DIR}")

set(CUDA_ARCH_BIN "30 35 50 52 61" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")

add_subdirectory(src)
add_subdirectory(test)