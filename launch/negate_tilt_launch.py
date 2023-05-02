import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("cam_lidar_bb_regression_cuda"),
        "config",
        "params.yaml",
    )
    """Generate launch description with multiple components."""
    container = ComposableNodeContainer(
        name="projections_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="cam_lidar_bb_regression_cuda",
                plugin="PerceptionNS::BbRegressorCuda",
                name="bb_regressor_cuda_exe",
                remappings=[
                    # ("input/cam", "/baumer/sf_stereo_right/image_raw"),
                    # ("/input/camera_info", "/baumer/sf_stereo_right/camera_info"),
                    # ("input/pointcloud", "/velodyne_points"),
                    ("input/cam", "/out/image_raw"),
                    ("/input/camera_info", "/out/camera_info"),
                    ("input/pointcloud", "/topicmaker/pcl_points"),
                    ("input/roi", "/rois"),
                ],
                parameters=[config],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
        output="both",
    )

    return launch.LaunchDescription([container])
