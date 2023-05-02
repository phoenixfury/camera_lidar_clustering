// Copyright 2023 Mostafa_Hegazy

#include "cam_lidar_bb_regression_cuda/node.hpp"

#include "cam_lidar_bb_regression_cuda/Matmul_kernels.hpp"
#include "cam_lidar_bb_regression_cuda/get_pointsInsideBbs_kernels.hpp"
#include "cam_lidar_bb_regression_cuda/utils.hpp"

#include <opencv2/core/eigen.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>
using autoware_auto_perception_msgs::msg::DetectedObject;
using autoware_auto_perception_msgs::msg::DetectedObjects;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
namespace PerceptionNS
{

static const rmw_qos_profile_t rmw_qos_profile_sensor = {
  rmw_qos_history_policy_t::RMW_QOS_POLICY_HISTORY_KEEP_LAST,
  1,
  rmw_qos_reliability_policy_t::RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
  rmw_qos_durability_policy_t::RMW_QOS_POLICY_DURABILITY_SYSTEM_DEFAULT,
  RMW_QOS_DEADLINE_DEFAULT,
  RMW_QOS_LIFESPAN_DEFAULT,
  rmw_qos_liveliness_policy_t::RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
  RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
  false};

BbRegressorCuda::BbRegressorCuda(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("bb_regressor_node", node_options),
  tf_buffer_(get_clock()),
  tf_listener_(tf_buffer_),
  m_video_qos(1)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  k_vector_ = this->declare_parameter<std::vector<double>>("k");
  d_vector_ = this->declare_parameter<std::vector<double>>("d");

  if (!check_matrices_validity(k_vector_, d_vector_, k_mat_, d_vec_)) {
    RCLCPP_ERROR(this->get_logger(), "Wrong intrinsic matrix size in param file");
  }
  use_cam_info_ = this->declare_parameter<bool>("use_cam_info");
  image_type_ = this->declare_parameter<std::string>("image_type");

  publish_clusters_ = this->declare_parameter<bool>("publish_clusters");
  publish_debug_image_ = this->declare_parameter<bool>("publish_debug_image");
  publish_detected_objects_ = this->declare_parameter<bool>("publish_detected_objects");

  lidar_name_ = this->declare_parameter<std::string>("lidar_name");
  cam_name_ = this->declare_parameter<std::string>("cam_name");

  nms_threshold_ = this->declare_parameter<double>("nms_threshold");
  crop_box_center_ = this->declare_parameter<std::vector<double>>("crop_box_center");
  crop_box_dims_ = this->declare_parameter<std::vector<double>>("crop_box_dims");
  filter_limits_ = this->declare_parameter<std::vector<double>>("filter_limits");

  m_video_qos.keep_last(2);
  m_video_qos.best_effort();
  m_video_qos.durability_volatile();

  img_sub_ = image_transport::create_camera_subscription(
    this, "input/cam", std::bind(&BbRegressorCuda::camera_callback, this, _1, _2), image_type_,
    m_video_qos.get_rmw_qos_profile());

  bb_sub_ = create_subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "input/roi", rclcpp::SensorDataQoS{}.keep_last(1),
    std::bind(&BbRegressorCuda::yolo_callback, this, std::placeholders::_1));

  pcl_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "input/pointcloud", rclcpp::SensorDataQoS{}.keep_last(1),
    std::bind(&BbRegressorCuda::pcl_callback, this, std::placeholders::_1));

  distribution_ = std::uniform_real_distribution<double>(0.0, 1.0);

  objects_pub_ =
    create_publisher<derived_object_msgs::msg::ObjectArray>("/out/detected_objs", rclcpp::QoS{1});

  img_pub_ =
    image_transport::create_publisher(this, "/out/debug_image", m_video_qos.get_rmw_qos_profile());

  pointcloud_pub_ =
    create_publisher<sensor_msgs::msg::PointCloud2>("out/clustered_pcl", rclcpp::SensorDataQoS{});
  p_mat_ = cv::Mat::zeros(cv::Size(3, 4), CV_32F);
  std::cout << "INIT";
}

void BbRegressorCuda::remove_outliers(
  std::shared_ptr<open3d::geometry::PointCloud> pcl_cloud_ptr, std::vector<int> & mean_points_vec,
  std::vector<std::vector<int>> & clusters_idcs, std::vector<cv::Rect> rects)
{
  for (size_t i = 0; i < mean_points_vec.size(); i++) {
    float b0 = -2;
    float b1 = 8;
    float rect_area_ratio = rects[i].area() / static_cast<float>(image_.cols * image_.rows);
    // std::cout << "h: " << rect_area_ratio << " " << -(b0 + b1 * rect_area_ratio) << std::endl;

    auto rect_ratio = 1 / (1 + exp(-(b0 + b1 * rect_area_ratio)));
    std::cout << rect_area_ratio << " " << rect_ratio << std::endl;
    for (int j = clusters_idcs[i].size() - 1; j >= 0; j--) {
      double x_mean = pcl_cloud_ptr->points_[mean_points_vec[i]].x();
      double y_mean = pcl_cloud_ptr->points_[mean_points_vec[i]].y();

      double x_point = pcl_cloud_ptr->points_[clusters_idcs[i][j]].x();
      double y_point = pcl_cloud_ptr->points_[clusters_idcs[i][j]].y();

      double d_mean = (x_mean * x_mean) + (y_mean * y_mean);
      double d_point = (x_point * x_point) + (y_point * y_point);
      float ratio = d_mean / d_point;

      if (ratio > 1 + rect_ratio || ratio < 1 - rect_ratio) {
        // if (ratio > filter_limits_[1] || ratio < filter_limits_[0]) {
        // std::cout << "x.y mean: " << x_mean << " " << y_mean << " point: " << x_point << " "
        //           << y_point << " d_mean: " << d_mean << " d_point: " << d_point
        //           << " ratio: " << ratio << std::endl;
        clusters_idcs[i].erase(clusters_idcs[i].begin() + j);
      }
    }
  }
}

void BbRegressorCuda::convert_boxes_to_objects(
  std::vector<std::shared_ptr<open3d::geometry::OrientedBoundingBox>> & boxes_vector,
  derived_object_msgs::msg::ObjectArray & objects_array)
{
  objects_array_.objects.clear();
  objects_array_.objects.reserve(boxes_vector.size());

  for (auto & box : boxes_vector) {
    derived_object_msgs::msg::Object der_object;
    der_object.pose.position.x = box->center_[0];
    der_object.pose.position.y = box->center_[1];
    der_object.pose.position.z = box->center_[2];

    const Eigen::Quaterniond bboxQ1(box->R_);

    Eigen::Vector4d q = bboxQ1.coeffs();

    der_object.pose.orientation.x = q[0];
    der_object.pose.orientation.y = q[1];
    der_object.pose.orientation.z = q[2];
    der_object.pose.orientation.w = q[3];

    der_object.shape.type = der_object.shape.BOX;
    der_object.shape.dimensions.emplace_back(box->extent_[0]);
    der_object.shape.dimensions.emplace_back(box->extent_[1]);
    der_object.shape.dimensions.emplace_back(box->extent_[2]);

    der_object.classification = derived_object_msgs::msg::Object::CLASSIFICATION_BOAT;

    der_object.object_classified = true;

    der_object.detection_level = der_object.OBJECT_DETECTED;

    objects_array.objects.emplace_back(der_object);
  }
}
void BbRegressorCuda::camera_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & input_img,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info)
{
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(input_img, sensor_msgs::image_encodings::BGR8);

    image_ = cv_ptr->image;
    if (use_cam_info_) {
      std::vector<double> k_vec(cam_info->k.begin(), cam_info->k.end());
      std::vector<double> p_vec(cam_info->p.begin(), cam_info->p.end());

      check_matrices_validity(k_vec, cam_info->d, p_vec, k_mat_, d_vec_, p_mat_);
    }
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(
      this->get_logger(), "Could not convert from '%s' to 'bgr8'.", input_img->encoding.c_str());
  }
}
void BbRegressorCuda::yolo_callback(
  const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr & input_roi_msg)
{
  bbs_.clear();
  auto objs_2d = input_roi_msg->feature_objects;
  bbs_.reserve(objs_2d.size());
  for (auto & ft_obj : objs_2d) {
    cv::Rect rect(
      ft_obj.feature.roi.x_offset, ft_obj.feature.roi.y_offset, ft_obj.feature.roi.width,
      ft_obj.feature.roi.height);
    bbs_.emplace_back(rect);
  }
}

void BbRegressorCuda::pcl_callback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_pointcloud_msg)
{
  auto t1 = high_resolution_clock::now();

  const auto ros_lidarToCam_world = get_transform(tf_buffer_, lidar_name_, cam_name_, this->now());
  transform_as_matrix(*ros_lidarToCam_world, BbRegressorCuda::tf_lidar_to_cam_);

  Eigen::MatrixXf k_eig_mat;
  Eigen::MatrixXf p_eig_mat = Eigen::MatrixXf::Zero(3, 4);
  // cv::Mat cv_mat(3, 4, CV_32FC1);
  // cv_mat = p_mat_;
  cv::cv2eigen(k_mat_, k_eig_mat);
  // for()
  // p_mat_.

  // std::cout << p_mat_.at<double>(0, 0) << " " << p_mat_.at<double>(0, 1) << " "
  //           << p_mat_.at<double>(0, 2) << " " << p_mat_.at<double>(0, 3) << " "
  //           << p_mat_.at<double>(1, 0) << " " << p_mat_.at<double>(1, 1) << " "
  //           << p_mat_.at<double>(1, 2) << " " << p_mat_.at<double>(1, 3) << " "
  //           << p_mat_.at<double>(2, 0) << " " << p_mat_.at<double>(2, 1) << " "
  //           << p_mat_.at<double>(2, 2) << " " << p_mat_.at<double>(2, 3) << " " << std::endl;
  p_eig_mat << p_mat_.at<double>(0, 0), p_mat_.at<double>(0, 1), p_mat_.at<double>(0, 2),
    p_mat_.at<double>(0, 3), p_mat_.at<double>(1, 0), p_mat_.at<double>(1, 1),
    p_mat_.at<double>(1, 2), p_mat_.at<double>(1, 3), p_mat_.at<double>(2, 0),
    p_mat_.at<double>(2, 1), p_mat_.at<double>(2, 2), p_mat_.at<double>(2, 3);
  // std::cout << p_mat_.rows << " " << p_mat_.cols << std::endl;
  // cv::cv2eigen(p_mat_.t(), p_eig_mat);
  // p_eig_mat.transposeInPlace();
  // auto x = p_eig_mat.transpose();
  // std::cout << p_mat_ << std::endl
  //           << tf_lidar_to_cam_ << std::endl
  //           << p_eig_mat << " " << p_eig_mat.rows() << " " << p_eig_mat.cols()
  //           << p_eig_mat * tf_lidar_to_cam_ << std::endl;

  // projection_mat_ = k_eig_mat * tf_lidar_to_cam_.topRows(3);
  projection_mat_ = p_eig_mat * tf_lidar_to_cam_;

  duration<double, std::milli> ms_double = high_resolution_clock::now() - t1;

  std::cout << " matrix time: " << ms_double.count() << std::endl;

  auto pcl_cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();

  ros_to_open3d(input_pointcloud_msg, *pcl_cloud_ptr);

  auto crop_bb = open3d::geometry::OrientedBoundingBox();
  Eigen::Vector3d eigen_crop_bb_center(
    crop_box_center_[0], crop_box_center_[1], crop_box_center_[2]),
    eigen_crop_bb_dims(crop_box_dims_[0], crop_box_dims_[1], crop_box_dims_[2]);
  crop_bb.center_ = eigen_crop_bb_center;
  crop_bb.extent_ = eigen_crop_bb_dims;

  auto forward_pointcloud_ptr =
    std::make_shared<open3d::geometry::PointCloud>(*pcl_cloud_ptr->Crop(crop_bb));

  // auto [coeffs, idcs] = forward_pointcloud_ptr->SegmentPlane();

  // forward_pointcloud_ptr = forward_pointcloud_ptr->SelectByIndex(idcs, true);
  std::vector<cv::Point2d> points_2d;
  std::vector<cv::Point3d> points_3d;

  t1 = high_resolution_clock::now();
  lidar_points_to_image_points(forward_pointcloud_ptr->points_, projection_mat_, points_2d);
  ms_double = high_resolution_clock::now() - t1;
  std::cout << " parallel time: " << ms_double.count() << std::endl;
  auto bbs = nms(bbs_, nms_threshold_);

  t1 = high_resolution_clock::now();
  auto clusters_idcs = PerceptionNS::get_points_in_bounding_boxes(points_2d, bbs);
  ms_double = high_resolution_clock::now() - t1;
  std::cout << " parallel 2 time: " << ms_double.count() << std::endl;

  for (int i = clusters_idcs.size() - 1; i >= 0; i--) {
    if (clusters_idcs[i].size() < 2) {
      clusters_idcs.erase(clusters_idcs.begin() + i);
      bbs.erase(bbs.begin() + i);
    }
  }

  t1 = high_resolution_clock::now();

  auto mean_points_vec = PerceptionNS::get_closest_points_to_centers(points_2d, bbs, clusters_idcs);
  ms_double = high_resolution_clock::now() - t1;
  std::cout << " parallel 3 time: " << ms_double.count() << std::endl;

  remove_outliers(forward_pointcloud_ptr, mean_points_vec, clusters_idcs, bbs);

  if (publish_debug_image_)
    for (auto & r : bbs) {
      cv::rectangle(image_, r, cv::Scalar(0, 0, 255));
      // std::cout << "area: " << r.area() << " "
      //           << " total area: " << image_.cols * image_.rows << std::endl;
      // std::cout << "y: " << r.br() << " "
      //           << " total y: " << image_.cols << std::endl;
      // std::cout << "center: " << r.x << " " << r.y << std::endl;
      for (auto & cluster : clusters_idcs) {
        for (auto & c_i : cluster) {
          cv::circle(image_, points_2d[c_i], 2, cv::Scalar(0, 255, 0), -1);
        }
      }
      std_msgs::msg::Header cam_header;
      cam_header.frame_id = cam_name_;

      cam_header.stamp = input_pointcloud_msg->header.stamp;
      sensor_msgs::msg::Image::SharedPtr img_msg =
        cv_bridge::CvImage(cam_header, "bgr8", image_).toImageMsg();

      img_pub_.publish(img_msg);
    }

  forward_pointcloud_ptr->colors_.resize(
    forward_pointcloud_ptr->points_.size(), Eigen::Vector3d(1, 1, 1));

  std::vector<std::shared_ptr<open3d::geometry::OrientedBoundingBox>> boxes;

  for (size_t i = 0; i < clusters_idcs.size(); i++) {
    boxes.emplace_back(std::make_shared<open3d::geometry::OrientedBoundingBox>());
    Eigen::Vector3d color;
    std::vector<size_t> sizet_vec(clusters_idcs[i].begin(), clusters_idcs[i].end());

    auto cluster = forward_pointcloud_ptr->SelectByIndex(sizet_vec);

    color << distribution_(generator_), distribution_(generator_), distribution_(generator_);
    *boxes[i] = regress_bounding_box(cluster);
    boxes[i]->color_ = color;

    for (auto & idx : clusters_idcs[i]) {
      forward_pointcloud_ptr->colors_[idx] = color;
    }
  }
  // Trial part
  for (size_t i = 0; i < bbs.size(); i++) {
    auto b_p = back_project(bbs[i].tl(), boxes[i]->center_[0], k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(0, 1, 0));

    b_p = back_project(bbs[i].br(), boxes[i]->center_[0], k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(0, 1, 0));

    b_p = back_project(
      cv::Point(bbs[i].x + bbs[i].width, bbs[i].y), boxes[i]->center_[0], k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(0, 1, 0));

    b_p = back_project(
      cv::Point(bbs[i].x, bbs[i].y + bbs[i].height), boxes[i]->center_[0], k_mat_,
      tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(0, 1, 0));

    b_p = back_project(
      bbs[i].tl(), boxes[i]->center_[0] - boxes[i]->extent_[0], k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(1, 0, 0));

    b_p = back_project(
      bbs[i].br(), boxes[i]->center_[0] - boxes[i]->extent_[0], k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(1, 0, 0));

    b_p = back_project(
      cv::Point(bbs[i].x + bbs[i].width, bbs[i].y), boxes[i]->center_[0] - boxes[i]->extent_[0],
      k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(1, 0, 0));

    b_p = back_project(
      cv::Point(bbs[i].x, bbs[i].y + bbs[i].height), boxes[i]->center_[0] - boxes[i]->extent_[0],
      k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(1, 0, 0));

    b_p = back_project(
      bbs[i].tl(), boxes[i]->center_[0] + boxes[i]->extent_[0], k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(0, 0, 1));

    b_p = back_project(
      bbs[i].br(), boxes[i]->center_[0] + boxes[i]->extent_[0], k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(0, 0, 1));

    b_p = back_project(
      cv::Point(bbs[i].x + bbs[i].width, bbs[i].y), boxes[i]->center_[0] + boxes[i]->extent_[0],
      k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(0, 0, 1));
    b_p = back_project(
      cv::Point(bbs[i].x, bbs[i].y + bbs[i].height), boxes[i]->center_[0] + boxes[i]->extent_[0],
      k_mat_, tf_lidar_to_cam_);

    forward_pointcloud_ptr->points_.emplace_back(b_p);
    forward_pointcloud_ptr->colors_.emplace_back(Eigen::Vector3d(0, 0, 1));
  }
  // End Trial part

  if (publish_detected_objects_) {
    objects_array_.header = input_pointcloud_msg->header;
    convert_boxes_to_objects(boxes, objects_array_);
    objects_pub_->publish(objects_array_);
  }

  if (publish_clusters_) {
    sensor_msgs::msg::PointCloud2 out_pcl;
    out_pcl.header = input_pointcloud_msg->header;
    open3d_to_ros(*forward_pointcloud_ptr, out_pcl);
    pointcloud_pub_->publish(out_pcl);
  }
}

void BbRegressorCuda::lidar_points_to_image_points(
  std::vector<Eigen::Vector3d> & input_cloud_points, Eigen::MatrixXf & projection_mat,
  std::vector<cv::Point2d> & out_cv_vec)

{
  auto input_size = input_cloud_points.size();
  Eigen::MatrixXf input_pcl_mat = Eigen::MatrixXf::Ones(4, input_size);

  input_pcl_mat.topRows(3) =
    Eigen::Map<Eigen::MatrixXd>(input_cloud_points[0].data(), 3, input_size).cast<float>();

  out_cv_vec.reserve(input_size);

  float * transform_vec_ = projection_mat.data();
  float * input_vec_ = input_pcl_mat.data();

  int m = projection_mat.rows();
  int n = input_pcl_mat.cols();
  int k = input_pcl_mat.rows();

  uint size_A = projection_mat.size();
  uint size_B = input_pcl_mat.size();
  uint size_C = m * n;

  float * host_C = (float *)malloc(sizeof(float) * size_C);

  cv::Point2d * host_cv_vec = (cv::Point2d *)malloc(sizeof(cv::Point2d) * n);

  PerceptionNS::project_3Dpts_to_2Dpts(
    transform_vec_, input_vec_, host_C, m, n, k, size_A, size_B, size_C, host_cv_vec);

  out_cv_vec.assign(host_cv_vec, host_cv_vec + n);

  free(host_C);
  free(host_cv_vec);
}

Eigen::Vector3d BbRegressorCuda::back_project(
  cv::Point point_2d, double depth, cv::Mat k, Eigen::Matrix4f projection_mat)
{
  geometry_msgs::msg::Point cam_3d_position;
  cam_3d_position.x = ((point_2d.x - k.at<double>(0, 2)) * depth) / k.at<double>(0, 0);
  cam_3d_position.y = ((point_2d.y - k.at<double>(1, 2)) * depth) / k.at<double>(1, 1);
  cam_3d_position.z = depth;

  Eigen::Vector4f point;
  point << cam_3d_position.x, cam_3d_position.y, cam_3d_position.z, 1;

  // Transform back to lidar's frame
  auto point_transformed = projection_mat.inverse() * point;
  return point_transformed.topRows(3).cast<double>();
}
open3d::geometry::OrientedBoundingBox BbRegressorCuda::regress_bounding_box(
  std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr)
{
  std::vector<MinimalBoundingBoxNS::MinimalBoundingBox::Point> points_vec(
    cloud_ptr->points_.size());

  double min_z = cloud_ptr->points_[0].z();
  double max_z = cloud_ptr->points_[0].z();
  for (size_t j = 0; j < cloud_ptr->points_.size(); j++) {
    MinimalBoundingBoxNS::MinimalBoundingBox::Point test_point;

    test_point.x = cloud_ptr->points_[j].x();
    test_point.y = cloud_ptr->points_[j].y();
    if (cloud_ptr->points_[j].z() < min_z) {
      min_z = cloud_ptr->points_[j].z();
    }
    if (cloud_ptr->points_[j].z() > max_z) {
      max_z = cloud_ptr->points_[j].z();
    }
    points_vec[j] = test_point;
  }

  MinimalBoundingBoxNS::MinimalBoundingBox::BoundingBox bounding_box;
  MinimalBoundingBoxNS::MinimalBoundingBox box;
  bounding_box = box.calculate(points_vec);

  Eigen::Vector3d bb_center, bb_extent, bb_rot;

  bb_center << bounding_box.center.x, bounding_box.center.y, (max_z + min_z) / 2.;
  bb_extent << bounding_box.height, bounding_box.width, (max_z - min_z);
  bb_rot << -bounding_box.heightAngle, 0, 0;

  auto open3d_bounding_box = open3d::geometry::OrientedBoundingBox();

  open3d_bounding_box.center_ = bb_center;
  open3d_bounding_box.extent_ = bb_extent;
  open3d_bounding_box.R_ = open3d_bounding_box.GetRotationMatrixFromZYX(bb_rot);

  return open3d_bounding_box;
}
}  // namespace PerceptionNS

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(PerceptionNS::BbRegressorCuda)
