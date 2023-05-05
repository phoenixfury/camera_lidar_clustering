// Copyright 2023 Mostafa_Hegazy

#pragma once
#include "open3d/Open3D.h"

#include <Eigen/Dense>

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// TF
#include "camera_lidar_clustering/MinimalBoundingBox.hpp"
#include "camera_lidar_clustering/classic_nms.hpp"
#include "camera_lidar_clustering/open3d_conversions.hpp"

#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "derived_object_msgs/msg/object_array.hpp"
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace PerceptionNS
{

class CameraLidarClusterer : public rclcpp::Node
{
public:
  explicit CameraLidarClusterer(const rclcpp::NodeOptions & node_options);

private:
  // Global configuration variables
  bool use_cam_info_ = false;
  bool use_projection_mat_ = false;
  std::string image_type_ = "raw";

  bool publish_clusters_ = true;
  bool publish_debug_image_ = true;
  bool publish_detected_objects_ = true;

  double nms_threshold_ = 0.25;

  std::vector<double> crop_box_center_;
  std::vector<double> crop_box_dims_;
  std::vector<int64_t> sigmoid_coeffs_;

  cv::Mat k_mat_;
  cv::Mat d_vec_;
  cv::Mat p_mat_;
  std::vector<double> k_vector_;
  std::vector<double> d_vector_;

  std::string lidar_name_;
  std::string cam_name_;

  // Transformation functions and variables
  Eigen::Matrix4f tf_lidar_to_cam_;

  Eigen::MatrixXf projection_mat_;

  std::unordered_map<uint8_t, uint8_t> label_converter_;
  std::vector<uint8_t> classification_vector_;

  derived_object_msgs::msg::ObjectArray objects_array_;

  rclcpp::QoS m_video_qos;

  // TF variables
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
  rclcpp::Publisher<derived_object_msgs::msg::ObjectArray>::SharedPtr objects_pub_;
  image_transport::Publisher img_pub_;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_sub_;
  image_transport::CameraSubscriber img_sub_;
  rclcpp::Subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr bb_sub_;

  std::vector<cv::Rect> bbs_;
  cv::Mat image_;
  std::default_random_engine generator_;
  std::uniform_real_distribution<double> distribution_;

  void camera_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & input_img_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & input_cam_info_msg);

  void pcl_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_pointcloud_msg);

  void yolo_callback(
    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr & input_roi_msg);

  void lidar_points_to_image_points(
    std::vector<Eigen::Vector3d> & input_cloud_points, Eigen::MatrixXf & projection_mat,
    std::vector<cv::Point2d> & out_cv_vec);

  void convert_boxes_to_objects(
    std::vector<std::shared_ptr<open3d::geometry::OrientedBoundingBox>> & boxes_vector,
    derived_object_msgs::msg::ObjectArray & objects_array,
    std::vector<uint8_t> & classification_vec);

  void remove_outliers(
    std::shared_ptr<open3d::geometry::PointCloud> pcl_cloud_ptr, std::vector<int> & mean_points_vec,
    std::vector<std::vector<int>> & clusters_idcs, std::vector<cv::Rect> rects);

  open3d::geometry::OrientedBoundingBox regress_bounding_box(
    std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr);

  Eigen::Vector3d back_project(
    cv::Point point_2d, double depth, cv::Mat k, Eigen::Matrix4f projection_mat);
};

}  // namespace PerceptionNS
