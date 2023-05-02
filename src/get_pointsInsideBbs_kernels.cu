#include "cam_lidar_bb_regression_cuda/get_pointsInsideBbs_kernels.hpp"

namespace PerceptionNS
{

const int THREADS_PER_BLOCK = 128;

__device__ bool is_point_inside_box(const cv::Point2d & p, const cv::Rect & bounding_box)
{
  return (
    p.x >= bounding_box.x && p.x <= (bounding_box.x + bounding_box.width) &&
    p.y >= bounding_box.y && p.y <= bounding_box.y + bounding_box.height);
}

__global__ void get_points_inside_box_kernel(
  const cv::Point2d * points, const cv::Rect box, const int num_points, int * indices,
  int * num_points_inside_box)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < num_points) {
    if (is_point_inside_box(points[idx], box)) {
      int index = atomicAdd(num_points_inside_box, 1);
      indices[index] = idx;
    }
  }
}

__device__ uint get_distance_squared(const cv::Point2d & p, const cv::Point2d & center)
{
  return ((p.x - center.x) * (p.x - center.x) + (p.y - center.y) * (p.y - center.y));
}

__device__ uint get_distance_squared(const double & x, const double & y, const cv::Point2d & center)
{
  return ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y));
}

__global__ void reduce_closest_points_kernel(
  cv::Point2d * s_closestPoints, cv::Point2d * closestPoint, const cv::Point2d center,
  int numBlocks, int * thread_idx, int * min_idx)
{
  __shared__ cv::Point2d s_tempClosestPoints[1];

  // Reduce results within each block
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    auto dist_i_p1 = get_distance_squared(s_closestPoints[threadIdx.x + i], center);
    auto dist_i = get_distance_squared(s_closestPoints[threadIdx.x], center);

    if (threadIdx.x < i && dist_i_p1 < dist_i) {
      s_closestPoints[threadIdx.x].x = s_closestPoints[threadIdx.x + i].x;
      s_closestPoints[threadIdx.x].y = s_closestPoints[threadIdx.x + i].y;
      thread_idx[threadIdx.x] = thread_idx[threadIdx.x + i];
    }
    __syncthreads();
  }

  // Have one thread in block 0 write the result to global memory
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    closestPoint->x = s_closestPoints[0].x;
    closestPoint->y = s_closestPoints[0].y;
    *min_idx = thread_idx[0];
  }
}

// The first kernel, get_closest_point_kernel, each block writes its closest point to the
// corresponding index in closestPoint. The second kernel, reduce_closest_points_kernel, takes in
// the array of closest points for each block and reduces them to a single closest point. Each
// thread copies one closest point from the array into shared memory, and then the closest points
// are reduced within shared memory using the same technique as before. Finally, one thread writes
// the final result to global memory.*/

__global__ void get_closest_point_kernel(
  const cv::Point2d * points, int numPoints, const cv::Point2d center, cv::Point2d * closestPoint,
  int * thread_idx)
{
  __shared__ cv::Point2d s_closestPoints[THREADS_PER_BLOCK];
  __shared__ int s_indicesClosestPoints[THREADS_PER_BLOCK];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  int minDistSq = INT_MAX;
  double closest_x = DBL_MAX;
  double closest_y = DBL_MAX;

  while (tid < numPoints) {
    int distSq = get_distance_squared(points[tid], center);

    if (distSq < minDistSq) {
      minDistSq = distSq;
      closest_x = points[tid].x;
      closest_y = points[tid].y;
      s_indicesClosestPoints[threadIdx.x] = tid;
    }
    tid += blockDim.x * gridDim.x;
  }
  s_closestPoints[threadIdx.x].x = closest_x;
  s_closestPoints[threadIdx.x].y = closest_y;
  __syncthreads();

  // Reduce results within each block
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    auto dist_i_p1 = get_distance_squared(s_closestPoints[threadIdx.x + i], center);
    auto dist_i = get_distance_squared(s_closestPoints[threadIdx.x], center);

    if (threadIdx.x < i && dist_i_p1 < dist_i) {
      s_closestPoints[threadIdx.x].x = s_closestPoints[threadIdx.x + i].x;
      s_closestPoints[threadIdx.x].y = s_closestPoints[threadIdx.x + i].y;
      s_indicesClosestPoints[threadIdx.x] = s_indicesClosestPoints[threadIdx.x + i];
    }
    __syncthreads();
  }

  // Have one thread in each block write the result to global memory
  if (threadIdx.x == 0) {
    closestPoint[blockIdx.x].x = s_closestPoints[0].x;
    closestPoint[blockIdx.x].y = s_closestPoints[0].y;
    thread_idx[blockIdx.x] = s_indicesClosestPoints[0];
  }
}

int get_closest_point_to_center(const std::vector<cv::Point2d> & points, const cv::Rect & box)
{
  int num_points = points.size();

  int numBlocks = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  cv::Point2d *d_points, *d_closest_points;
  int * d_thread_idcs;

  checkCudaErrors(cudaMalloc((void **)&d_thread_idcs, numBlocks * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_points, num_points * sizeof(cv::Point2d)));

  checkCudaErrors(
    cudaMemcpy(d_points, points.data(), num_points * sizeof(cv::Point2d), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void **)&d_closest_points, numBlocks * sizeof(cv::Point2d)));

  cv::Point2d center((box.x + box.width / 2), (box.y + box.height / 2));

  get_closest_point_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(
    d_points, num_points, center, d_closest_points, d_thread_idcs);

  // Perform the reduction on the device
  cv::Point2d * d_closestPoint;
  int * d_min_idx;
  int min_idx;

  checkCudaErrors(cudaMalloc((void **)&d_closestPoint, 1 * sizeof(cv::Point2d)));
  checkCudaErrors(cudaMalloc((void **)&d_min_idx, 1 * sizeof(int)));

  reduce_closest_points_kernel<<<1, numBlocks>>>(
    d_closest_points, d_closestPoint, center, numBlocks, d_thread_idcs, d_min_idx);
  cv::Point2d h_closestPoint;

  checkCudaErrors(
    cudaMemcpy(&h_closestPoint, d_closestPoint, sizeof(cv::Point2d), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&min_idx, d_min_idx, sizeof(int), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_closestPoint));
  checkCudaErrors(cudaFree(d_points));
  checkCudaErrors(cudaFree(d_closest_points));
  checkCudaErrors(cudaFree(d_thread_idcs));
  checkCudaErrors(cudaFree(d_min_idx));

  return min_idx;
}

std::vector<int> get_closest_points_to_centers(
  const std::vector<cv::Point2d> & points, const std::vector<cv::Rect> & boxes,
  std::vector<std::vector<int>> & indices_vec)
{
  std::vector<int> points_in_boxes(boxes.size());

  for (int i = 0; i < indices_vec.size(); i++) {
    std::vector<cv::Point2d> image_points_vec;
    image_points_vec.reserve(indices_vec[i].size());

    for (auto & idx : indices_vec[i]) {
      image_points_vec.emplace_back(points[idx]);
    }
    auto closest_idx = get_closest_point_to_center(image_points_vec, boxes[i]);

    points_in_boxes[i] = indices_vec[i][closest_idx];
  }
  return points_in_boxes;
}
std::vector<int> get_points_inside_box(
  const std::vector<cv::Point2d> & points, const cv::Rect & box)
{
  int num_points = points.size();

  cv::Point2d * d_points;
  checkCudaErrors(cudaMalloc((void **)&d_points, num_points * sizeof(cv::Point2d)));
  checkCudaErrors(
    cudaMemcpy(d_points, points.data(), num_points * sizeof(cv::Point2d), cudaMemcpyHostToDevice));

  int * d_indices;
  checkCudaErrors(cudaMalloc((void **)&d_indices, num_points * sizeof(int)));

  int * d_num_points_inside_box;
  checkCudaErrors(cudaMalloc((void **)&d_num_points_inside_box, sizeof(int)));
  checkCudaErrors(cudaMemset(d_num_points_inside_box, 0, sizeof(int)));

  const int block_size = 256;
  const int num_blocks = (num_points + block_size - 1) / block_size;

  get_points_inside_box_kernel<<<num_blocks, block_size>>>(
    d_points, box, num_points, d_indices, d_num_points_inside_box);

  cudaDeviceSynchronize();

  int num_points_inside_box;
  checkCudaErrors(cudaMemcpy(
    &num_points_inside_box, d_num_points_inside_box, sizeof(int), cudaMemcpyDeviceToHost));

  std::vector<int> indices(num_points_inside_box);
  checkCudaErrors(cudaMemcpy(
    indices.data(), d_indices, num_points_inside_box * sizeof(int), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_points));
  checkCudaErrors(cudaFree(d_indices));
  checkCudaErrors(cudaFree(d_num_points_inside_box));

  return indices;
}

std::vector<std::vector<int>> get_points_in_bounding_boxes(
  const std::vector<cv::Point2d> & points, const std::vector<cv::Rect> & boxes)
{
  std::vector<std::vector<int>> points_in_boxes(boxes.size());

  for (int i = 0; i < boxes.size(); i++) {
    points_in_boxes[i] = get_points_inside_box(points, boxes[i]);
  }

  return points_in_boxes;
}
}  // namespace PerceptionNS
