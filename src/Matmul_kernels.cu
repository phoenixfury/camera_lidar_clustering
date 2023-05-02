// Copyright 2023 Mostafa_Hegazy

#include "cam_lidar_bb_regression_cuda/Matmul_kernels.hpp"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

namespace PerceptionNS
{

__global__ void projection_kernel(int cv_vec_n, float * pts_vec, cv::Point2d * cv_vec)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int s = 3;

  for (int i = index; i < cv_vec_n; i = i + stride) {
    auto denominator = 1 / pts_vec[i * s + 2];
    cv_vec[i].x = pts_vec[i * s] * denominator;
    cv_vec[i].y = pts_vec[i * s + 1] * denominator;
  }
}

void project_3Dpts_to_2Dpts(
  float * host_A, float * host_B, float * host_C, int m, int n, int k, uint size_A, uint size_B,
  uint size_C, cv::Point2d * h_cv_vec)
{
  int lda = m, ldb = k, ldc = m;
  // allocate host memory for matrices A and B
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int mem_size_B = sizeof(float) * size_B;

  // allocate device memory
  float * d_A;
  float * d_B;
  float * d_C;
  cv::Point2d * d_cv_vec;

  unsigned int mem_size_cv_vec = sizeof(cv::Point2d) * n;
  unsigned int mem_size_C = sizeof(float) * size_C;

  // allocate GPU memory for the matrices and copu from host to device
  checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
  checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
  checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));
  checkCudaErrors(cudaMalloc((void **)&d_cv_vec, mem_size_cv_vec));

  checkCudaErrors(cudaMemcpy(d_A, host_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, host_B, mem_size_B, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  checkCudaErrors(cublasSgemm(
    handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
  cudaDeviceSynchronize();

  int blockSize = 128;
  int numBlocks = (n + blockSize - 1) / blockSize;

  projection_kernel<<<numBlocks, blockSize>>>(n, d_C, d_cv_vec);

  checkCudaErrors(cudaMemcpy(h_cv_vec, d_cv_vec, mem_size_cv_vec, cudaMemcpyDeviceToHost));
  checkCudaErrors(cublasDestroy(handle));

  // clean up memory
  checkCudaErrors(cudaFree(d_cv_vec));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
}

void matrix_mul_cuda(
  float * host_A, float * host_B, float * host_C, int m, int n, int k, uint size_A, uint size_B,
  uint size_C)
{
  int lda = m, ldb = k, ldc = m;

  // allocate host memory for matrices A and B
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int mem_size_B = sizeof(float) * size_B;

  // allocate device memory
  float * d_A;
  float * d_B;
  float * d_C;

  unsigned int mem_size_C = sizeof(float) * size_C;

  // allocate GPU memory for the matrices and copu from host to device
  checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
  checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
  checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

  checkCudaErrors(cudaMemcpy(d_A, host_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, host_B, mem_size_B, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  checkCudaErrors(cublasSgemm(
    handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

  // copy result from device to host
  checkCudaErrors(cudaMemcpy(host_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  checkCudaErrors(cublasDestroy(handle));

  // clean up memory
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
}
}  // namespace PerceptionNS
