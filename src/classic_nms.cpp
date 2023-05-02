// Copyright 2016 Martin Kersner, m.kersner@gmail.com
// C++ version of http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

#include "cam_lidar_bb_regression_cuda/classic_nms.hpp"

using cv::Point;
using cv::Rect;
using std::vector;

namespace PerceptionNS
{

vector<Rect> nms(const vector<Rect> & boxes, const float & threshold)
{
  if (boxes.empty()) {
    return vector<Rect>();
  }

  // grab the coordinates of the bounding boxes
  auto points_vec = get_point_from_rect(boxes);

  auto x1 = points_vec[0];
  auto y1 = points_vec[1];
  auto x2 = points_vec[2];
  auto y2 = points_vec[3];

  // compute the area of the bounding boxes and sort the bounding
  // boxes by the bottom-right y-coordinate of the bounding box
  auto area = compute_area(x1, y1, x2, y2);
  auto idxs = argsort(y2);

  int last;
  int i;
  vector<int> pick;

  // keep looping while some indexes still remain in the indexes list
  while (idxs.size() > 0) {
    // grab the last index in the indexes list and add the
    // index value to the list of picked indexes
    last = idxs.size() - 1;
    i = idxs[last];
    pick.push_back(i);

    // find the largest (x, y) coordinates for the start of
    // the bounding box and the smallest (x, y) coordinates
    // for the end of the bounding box
    auto idxsWoLast = remove_last(idxs);

    auto xx1 = maximum(x1[i], copy_by_indexes(x1, idxsWoLast));
    auto yy1 = maximum(y1[i], copy_by_indexes(y1, idxsWoLast));
    auto xx2 = minimum(x2[i], copy_by_indexes(x2, idxsWoLast));
    auto yy2 = minimum(y2[i], copy_by_indexes(y2, idxsWoLast));

    // compute the width and height of the bounding box
    auto w = maximum(0, subtract(xx2, xx1));
    auto h = maximum(0, subtract(yy2, yy1));

    // compute the ratio of overlap
    auto overlap = divide(multiply(w, h), copy_by_indexes(area, idxsWoLast));

    // delete all indexes from the index list that have
    auto deleteIdxs = where_larger(overlap, threshold);
    deleteIdxs.push_back(last);

    idxs = remove_by_indexes(idxs, deleteIdxs);
  }

  return filter_vector(boxes, pick);
}

vector<vector<float>> get_point_from_rect(const vector<Rect> & rect)
{
  vector<float> points_x1, points_y1, points_x2, points_y2;
  vector<vector<float>> points_vec(4);
  points_x1.reserve(rect.size());
  points_y1.reserve(rect.size());
  points_x2.reserve(rect.size());
  points_y2.reserve(rect.size());

  for (const auto & p : rect) {
    points_x1.push_back(p.x);
    points_y1.push_back(p.y);
    points_x2.push_back(p.x + p.width);
    points_y2.push_back(p.y + p.height);
  }
  points_vec[0] = points_x1;
  points_vec[1] = points_y1;
  points_vec[2] = points_x2;
  points_vec[3] = points_y2;

  return points_vec;
}

vector<float> compute_area(
  const vector<float> & x1, const vector<float> & y1, const vector<float> & x2,
  const vector<float> & y2)
{
  vector<float> area;
  auto len = x1.size();

  for (decltype(len) idx = 0; idx < len; ++idx) {
    auto tmpArea = (x2[idx] - x1[idx] + 1) * (y2[idx] - y1[idx] + 1);
    area.push_back(tmpArea);
  }

  return area;
}

template <typename T>
vector<int> argsort(const vector<T> & v)
{
  // initialize original index locations
  vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] < v[i2]; });

  return idx;
}

vector<float> maximum(const float & num, const vector<float> & vec)
{
  auto maxVec = vec;
  auto len = vec.size();

  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] < num) maxVec[idx] = num;

  return maxVec;
}

vector<float> minimum(const float & num, const vector<float> & vec)
{
  auto minVec = vec;
  auto len = vec.size();

  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] > num) minVec[idx] = num;

  return minVec;
}

vector<float> copy_by_indexes(const vector<float> & vec, const vector<int> & idxs)
{
  vector<float> resultVec;

  for (const auto & idx : idxs) resultVec.push_back(vec[idx]);

  return resultVec;
}

vector<int> remove_last(const vector<int> & vec)
{
  auto resultVec = vec;
  resultVec.erase(resultVec.end() - 1);
  return resultVec;
}

vector<float> subtract(const vector<float> & vec1, const vector<float> & vec2)
{
  vector<float> result;
  auto len = vec1.size();

  for (decltype(len) idx = 0; idx < len; ++idx) result.push_back(vec1[idx] - vec2[idx] + 1);

  return result;
}

vector<float> multiply(const vector<float> & vec1, const vector<float> & vec2)
{
  vector<float> resultVec;
  auto len = vec1.size();

  for (decltype(len) idx = 0; idx < len; ++idx) resultVec.push_back(vec1[idx] * vec2[idx]);

  return resultVec;
}

vector<float> divide(const vector<float> & vec1, const vector<float> & vec2)
{
  vector<float> resultVec;
  auto len = vec1.size();

  for (decltype(len) idx = 0; idx < len; ++idx) resultVec.push_back(vec1[idx] / vec2[idx]);

  return resultVec;
}

vector<int> where_larger(const vector<float> & vec, const float & threshold)
{
  vector<int> resultVec;
  auto len = vec.size();

  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] > threshold) resultVec.push_back(idx);

  return resultVec;
}

vector<int> remove_by_indexes(const vector<int> & vec, const vector<int> & idxs)
{
  auto resultVec = vec;
  auto offset = 0;

  for (const auto & idx : idxs) {
    resultVec.erase(resultVec.begin() + idx + offset);
    offset -= 1;
  }

  return resultVec;
}

vector<Rect> boxes_to_rectangles(const vector<vector<float>> & boxes)
{
  vector<Rect> rectangles;
  vector<float> box;

  for (const auto & box : boxes)
    rectangles.push_back(Rect(Point(box[0], box[1]), Point(box[2], box[3])));

  return rectangles;
}

template <typename T>
vector<T> filter_vector(const vector<T> & vec, const vector<int> & idxs)
{
  vector<T> resultVec;

  for (const auto & idx : idxs) resultVec.push_back(vec[idx]);

  return resultVec;
}

}  // namespace PerceptionNS
