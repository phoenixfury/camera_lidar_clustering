// Copyright 2022 Mostafa_Hegazy

#include "open3d/Open3D.h"

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <string>
/**
 * @brief Copy data from a open3d::geometry::PointCloud to a sensor_msgs::PointCloud2
 *
 * @param pointcloud Reference to the open3d PointCloud
 * @param ros_pc2 Reference to the sensor_msgs PointCloud2
 * @param frame_id The string to be placed in the frame_id of the PointCloud2
 */
void open3d_to_ros(
  const open3d::geometry::PointCloud & pointcloud, sensor_msgs::msg::PointCloud2 & ros_pc2,
  std::string frame_id = "velodyne32");

/**
 * @brief Copy data from a sensor_msgs::msg::PointCloud2 to a open3d::geometry::PointCloud
 *
 * @param ros_pc2 Reference to the sensor_msgs PointCloud2
 * @param o3d_pc Reference to the open3d PointCloud
 * @param skip_colors If true, only xyz fields will be copied
 */

void ros_to_open3d(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & ros_pc2,
  open3d::geometry::PointCloud & o3d_pc, bool skip_colors = false);
/**
 *@brief Copy data from a open3d::t::geometry::PointCloud to a sensor_msgs::msg::PointCloud2
 *
 *@param pointcloud Reference to the open3d tgeometry PointCloud
 *@param ros_pc2 Reference to the sensor_msgs PointCloud2
 *@param frame_id The string to be placed in the frame_id of the PointCloud2
 *@param t_num_fields Twice the number of fields that the pointcloud contains
 *@param var_args Strings of field names followed succeeded by their datatype ("int" / "float")
 */
void open3d_to_ros(
  const open3d::t::geometry::PointCloud & pointcloud, sensor_msgs::msg::PointCloud2 & ros_pc2,
  std::string frame_id = "velodyne32", int t_num_fields = 2, ...);

/**
 * @brief Copy data from a sensor_msgs::msg::PointCloud2 to a open3d::t::geometry::PointCloud
 *
 * @param ros_pc2 Reference to the sensor_msgs PointCloud2
 * @param o3d_pc Reference to the open3d tgeometry PointCloud
 * @param skip_colors If true, only xyz fields will be copied
 */
void ros_to_open3d(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & ros_pc2,
  open3d::t::geometry::PointCloud & o3d_pc, bool skip_colors = false);
