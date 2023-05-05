// Copyright 2023 Mostafa_Hegazy

#pragma once

// CUDA runtime
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA utility functions
#include "camera_lidar_clustering/cuda_utils.hpp"

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>

namespace PerceptionNS
{

void matrix_mul_cuda(
  float * host_A, float * host_B, float * host_C, int m, int n, int k, uint size_A, uint size_B,
  uint size_C);

void project_3Dpts_to_2Dpts(
  float * host_A, float * host_B, float * host_C, int m, int n, int k, uint size_A, uint size_B,
  uint size_C, cv::Point2d * h_cv_vec);

}  // namespace PerceptionNS
