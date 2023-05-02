// Copyright 2023 Mostafa_Hegazy

#pragma once
#include <Eigen/Dense>

// C++
#include "open3d/Open3D.h"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <memory>
#include <string>
#include <thread>
#include <vector>

// Synchro
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// TF
#include "cam_lidar_bb_regression_cuda/MinimalBoundingBox.hpp"
#include "cam_lidar_bb_regression_cuda/classic_nms.hpp"
#include "cam_lidar_bb_regression_cuda/open3d_conversions.hpp"

#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "autoware_auto_perception_msgs/msg/detected_objects.hpp"
#include "derived_object_msgs/msg/object_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_interface.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/transform_listener.h>

namespace PerceptionNS
{

class BbRegressorCuda : public rclcpp::Node
{
public:
  explicit BbRegressorCuda(const rclcpp::NodeOptions & node_options);

private:
  // Global configuration variables
  bool use_cam_info_ = false;
  std::string image_type_ = "raw";

  bool publish_clusters_ = true;
  bool publish_debug_image_ = true;
  bool publish_detected_objects_ = true;

  double nms_threshold_ = 0.25;

  std::vector<double> crop_box_center_;
  std::vector<double> crop_box_dims_;
  std::vector<double> filter_limits_;

  cv::Mat k_mat_;
  cv::Mat d_vec_;
  cv::Mat p_mat_;
  std::vector<double> k_vector_;
  std::vector<double> d_vector_;

  std::string lidar_name_;
  std::string cam_name_;

  // Transformation functions and variables
  Eigen::Matrix4f tf_lidar_to_cam_;
  Eigen::Matrix4f tf_cam_to_lidar_;

  Eigen::MatrixXf projection_mat_;

  derived_object_msgs::msg::ObjectArray objects_array_;

  Eigen::Vector3d back_project(
    cv::Point point_2d, double depth, cv::Mat k, Eigen::Matrix4f projection_mat);

  // TF variables
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;

  // pcl subscriber
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_sub_;
  // image Subscriber
  image_transport::CameraSubscriber img_sub_;
  rclcpp::Subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr bb_sub_;

  std::vector<cv::Rect> bbs_;
  cv::Mat image_;
  std::default_random_engine generator_;
  std::uniform_real_distribution<double> distribution_;
  std::shared_ptr<open3d::geometry::Geometry> geo_vis;

  void camera_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & input_img_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & input_cam_info_msg);
  void pcl_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_pointcloud_msg);
  void yolo_callback(
    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr & input_roi_msg);
  void lidar_points_to_image_points(
    std::vector<Eigen::Vector3d> & input_cloud_points, Eigen::MatrixXf & projection_mat,
    std::vector<cv::Point2d> & out_cv_vec);

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub_;

  rclcpp::Publisher<derived_object_msgs::msg::ObjectArray>::SharedPtr objects_pub_;

  image_transport::Publisher img_pub_;
  rclcpp::QoS m_video_qos;

  void convert_boxes_to_objects(
    std::vector<std::shared_ptr<open3d::geometry::OrientedBoundingBox>> & boxes_vector,
    derived_object_msgs::msg::ObjectArray & objects_array);

  void remove_outliers(
    std::shared_ptr<open3d::geometry::PointCloud> pcl_cloud_ptr, std::vector<int> & mean_points_vec,
    std::vector<std::vector<int>> & clusters_idcs, std::vector<cv::Rect> rects);

  open3d::geometry::OrientedBoundingBox regress_bounding_box(
    std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr);
};

}  // namespace PerceptionNS
