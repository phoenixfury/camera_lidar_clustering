// Copyright 2022 Mostafa_Hegazy

#include "camera_lidar_clustering/utils.hpp"

#include <string>

using std::string;

namespace PerceptionNS
{
/**
 * @brief Get the Transformation between 2 frames
 *
 * @param tf_buffer
 * @param source_frame_id
 * @param target_frame_id
 * @param time
 * @param str
 * @return boost::optional<geometry_msgs::msg::TransformStamped>
 */
std::optional<geometry_msgs::msg::TransformStamped> get_transform(
  const tf2_ros::Buffer & tf_buffer, const string & source_frame_id, const string & target_frame_id,
  const rclcpp::Time & time)
{
  try {
    geometry_msgs::msg::TransformStamped self_transform_stamped;
    self_transform_stamped = tf_buffer.lookupTransform(
      /*target*/ target_frame_id, /*src*/ source_frame_id, time,
      rclcpp::Duration::from_seconds(0.5));
    return self_transform_stamped;
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("negate_tilt_node"), ex.what());
    return {};
  }
}

/**
 * @brief convert the transformation matrix into Eigen format
 *
 * @param bt tf2::Transform
 * @param out_mat Eigen::Matrix4f
 */
void transform_as_matrix(const tf2::Transform & bt, Eigen::Matrix4f & out_mat)
{
  double mv[12];
  bt.getBasis().getOpenGLSubMatrix(mv);

  tf2::Vector3 origin = bt.getOrigin();

  out_mat(0, 0) = mv[0];
  out_mat(0, 1) = mv[4];
  out_mat(0, 2) = mv[8];
  out_mat(1, 0) = mv[1];
  out_mat(1, 1) = mv[5];
  out_mat(1, 2) = mv[9];
  out_mat(2, 0) = mv[2];
  out_mat(2, 1) = mv[6];
  out_mat(2, 2) = mv[10];

  out_mat(3, 0) = out_mat(3, 1) = out_mat(3, 2) = 0;
  out_mat(3, 3) = 1;
  out_mat(0, 3) = origin.x();
  out_mat(1, 3) = origin.y();
  out_mat(2, 3) = origin.z();
}

bool check_matrices_validity(
  const std::vector<double> & k, const std::vector<double> & d, const std::vector<double> & p,
  cv::Mat & k_mat, cv::Mat & d_vec, cv::Mat & p_mat)
{
  if (k.size() != 9 || d.size() != 5 || p.size() != 12) {
    return false;
  }
  d_vec = cv::Mat1d(d, true);
  k_mat = cv::Mat(k, true).reshape(0, 3);
  p_mat = cv::Mat(p, true).reshape(0, 3);
  return true;
}
bool check_matrices_validity(
  const std::vector<double> & k, const std::vector<double> & d, cv::Mat & k_mat, cv::Mat & d_vec)
{
  if (k.size() != 9 || d.size() != 5) {
    return false;
  }
  d_vec = cv::Mat1d(d, true);
  k_mat = cv::Mat(k, true).reshape(0, 3);
  return true;
}
void transform_as_matrix(const geometry_msgs::msg::TransformStamped & bt, Eigen::Matrix4f & out_mat)
{
  tf2::Transform transform;
  tf2::convert(bt.transform, transform);
  transform_as_matrix(transform, out_mat);
}
}  // namespace PerceptionNS
