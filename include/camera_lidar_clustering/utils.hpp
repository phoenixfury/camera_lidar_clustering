// Copyright 2022 Mostafa_Hegazy

#pragma once

#include "rclcpp/logging.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <vector>

// TF
#include <tf2/LinearMath/Transform.h>
#include <tf2/convert.h>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <functional>
#include <optional>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "autoware_auto_perception_msgs/msg/detected_objects.hpp"
#include "autoware_auto_perception_msgs/msg/tracked_objects.hpp"
#include "derived_object_msgs/msg/object_array.hpp"

#include <string>
namespace PerceptionNS
{

  std::optional<geometry_msgs::msg::TransformStamped> get_transform(
      const tf2_ros::Buffer &tf_buffer, const std::string &source_frame_id,
      const std::string &target_frame_id, const rclcpp::Time &time);

  void transform_as_matrix(const tf2::Transform &bt, Eigen::Matrix4f &out_mat);
  bool check_matrices_validity(
      const std::vector<double> &k, const std::vector<double> &d, const std::vector<double> &p,
      cv::Mat &k_mat, cv::Mat &d_vec, cv::Mat &p_mat);
  bool check_matrices_validity(
      const std::vector<double> &k, const std::vector<double> &d, cv::Mat &k_mat, cv::Mat &d_vec);
  void transform_as_matrix(
      const geometry_msgs::msg::TransformStamped &bt, Eigen::Matrix4f &out_mat);
  autoware_auto_perception_msgs::msg::DetectedObjects derived_obj_msg_to_autoware_msg(
      derived_object_msgs::msg::ObjectArray der_objs);
} // namespace PerceptionNS
