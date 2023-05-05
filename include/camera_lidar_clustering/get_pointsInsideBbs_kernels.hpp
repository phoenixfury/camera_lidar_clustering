// Copyright 2023 Mostafa_Hegazy

#pragma once

// CUDA runtime
#include <cuda_runtime.h>

#include <vector>

// CUDA utility functions
#include "camera_lidar_clustering/cuda_utils.hpp"

#include <opencv2/core.hpp>

namespace PerceptionNS
{

std::vector<int> get_points_inside_box(
  const std::vector<cv::Point2d> & points, const cv::Rect & box);

std::vector<std::vector<int>> get_points_in_bounding_boxes(
  const std::vector<cv::Point2d> & points, const std::vector<cv::Rect> & boxes);

std::vector<int> get_closest_points_to_centers(
  const std::vector<cv::Point2d> & points, const std::vector<cv::Rect> & boxes,
  std::vector<std::vector<int>> & indices_vec);

int get_closest_point_to_center(const std::vector<cv::Point2d> & points, const cv::Rect & box);
}  // namespace PerceptionNS
