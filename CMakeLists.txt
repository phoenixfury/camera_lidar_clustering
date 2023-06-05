cmake_minimum_required(VERSION 3.5)
project(camera_lidar_clustering)

### Compile options
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(CUDA)
if(CUDA_FOUND)
  find_library(CUBLAS_LIBRARIES cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
  )
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND")
  set(CUDA_AVAIL OFF)
endif()

find_package(Open3D REQUIRED)

find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Open3D_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

ament_auto_add_library(camera_lidar_clustering SHARED
src/node.cpp
src/open3d_conversions.cpp
src/classic_nms.cpp
src/MinimalBoundingBox.cpp
utils/utils.cpp
)

cuda_add_library(cuda_kernels_lib SHARED
src/Matmul_kernels.cu
src/get_pointsInsideBbs_kernels.cu
)

target_link_libraries(camera_lidar_clustering
 ${OpenCV_LIBS}
${rclcpp_LIBRARIES}
${Open3D_LIBRARIES}
 ${CUDA_LIBRARIES}
${CUBLAS_LIBRARIES}
cuda_kernels_lib)

rclcpp_components_register_node(camera_lidar_clustering
  PLUGIN "PerceptionNS::CameraLidarClusterer"
  EXECUTABLE clusterer_cuda_exe
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
