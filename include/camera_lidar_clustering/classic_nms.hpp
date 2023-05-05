// Copyright 2016 Martin Kersner, m.kersner@gmail.com

#ifndef CLASSIC_NMS_HPP_INCLUDED
#define CLASSIC_NMS_HPP_INCLUDED

#include <opencv2/opencv.hpp>

#include <numeric>
#include <tuple>
#include <vector>

namespace PerceptionNS
{
enum PointInRectangle { XMIN, YMIN, XMAX, YMAX };

std::tuple<std::vector<cv::Rect>, std::vector<int>> nms(
  const std::vector<cv::Rect> &, const float &);

std::vector<std::vector<float>> get_point_from_rect(const std::vector<cv::Rect> &);

std::vector<float> compute_area(
  const std::vector<float> &, const std::vector<float> &, const std::vector<float> &,
  const std::vector<float> &);

template <typename T>
std::vector<int> argsort(const std::vector<T> & v);

std::vector<float> maximum(const float &, const std::vector<float> &);

std::vector<float> minimum(const float &, const std::vector<float> &);

std::vector<float> copy_by_indexes(const std::vector<float> &, const std::vector<int> &);

std::vector<int> remove_last(const std::vector<int> &);

std::vector<float> subtract(const std::vector<float> &, const std::vector<float> &);

std::vector<float> multiply(const std::vector<float> &, const std::vector<float> &);

std::vector<float> divide(const std::vector<float> &, const std::vector<float> &);

std::vector<int> where_larger(const std::vector<float> &, const float &);

std::vector<int> remove_by_indexes(const std::vector<int> &, const std::vector<int> &);

std::vector<cv::Rect> boxes_to_rectangles(const std::vector<std::vector<float>> &);

template <typename T>
std::vector<T> filter_vector(const std::vector<T> &, const std::vector<int> &);
}  // namespace PerceptionNS

#endif  // CLASSIC_NMS_HPP_INCLUDED
