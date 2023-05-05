// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 * This code is licensed under CC0 1.0 Universal (Public Domain).
 * You can use this without any limitation.
 * https://creativecommons.org/publicdomain/zero/1.0/deed.en
 */
#pragma once

#ifndef CUDA_UTILS_HPP_
#define CUDA_UTILS_HPP_

#include <cuda_runtime_api.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) (cuda::check((val), #val, __FILE__, __LINE__))

namespace cuda
{
// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char * _cudaGetErrorEnum(cudaError_t error)
{
  return cudaGetErrorName(error);
}
#endif

#ifdef CUDA_DRIVER_API
// CUDA Driver API errors
static const char * _cudaGetErrorEnum(CUresult error)
{
  static char unknown[] = "<unknown>";
  const char * ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char * _cudaGetErrorEnum(cublasStatus_t error)
{
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}
#endif
template <typename T>
void check(T result, char const * const func, const char * const file, int const line)
{
  if (result) {
    fprintf(
      stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
      static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

struct deleter
{
  void operator()(void * p) const { checkCudaErrors(::cudaFree(p)); }
};
template <typename T>
using unique_ptr = std::unique_ptr<T, deleter>;

template <typename T>
typename std::enable_if<std::is_array<T>::value, cuda::unique_ptr<T>>::type make_unique(
  const std::size_t n)
{
  using U = typename std::remove_extent<T>::type;
  U * p;
  checkCudaErrors(::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(U) * n));
  return cuda::unique_ptr<T>{p};
}

template <typename T>
cuda::unique_ptr<T> make_unique()
{
  T * p;
  checkCudaErrors(::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(T)));
  return cuda::unique_ptr<T>{p};
}

constexpr size_t CUDA_ALIGN = 256;

template <typename T>
inline size_t get_size_aligned(size_t num_elem)
{
  size_t size = num_elem * sizeof(T);
  size_t extra_align = 0;
  if (size % CUDA_ALIGN != 0) {
    extra_align = CUDA_ALIGN - size % CUDA_ALIGN;
  }
  return size + extra_align;
}

template <typename T>
inline T * get_next_ptr(size_t num_elem, void *& workspace, size_t & workspace_size)
{
  size_t size = get_size_aligned<T>(num_elem);
  if (size > workspace_size) {
    throw std::runtime_error("Workspace is too small!");
  }
  workspace_size -= size;
  T * ptr = reinterpret_cast<T *>(workspace);
  workspace = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(workspace) + size);
  return ptr;
}

}  // namespace cuda

#endif  // CUDA_UTILS_HPP_
