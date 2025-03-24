/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2023 Spotify AB
 *
 * This file is includes code from hnswlib (https://github.com/nmslib/hnswlib,
 * Apache 2.0-licensed, no copyright author listed)
 * --
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * -/-/-
 */

#pragma once
#include "Space.h"
#include "../simd_utils.h"
#include <ratio>

//#define USE_SIMD_DISPATCHER

namespace hnswlib {
/**
 * For a given loop unrolling factor K, distance type dist_t, and data type
 * data_t, calculate the L2 squared distance between two vectors. The compiler
 * should automatically do the loop unrolling for us here and vectorize as
 * appropriate.
 */
template <typename dist_t, typename data_t = dist_t, int K = 1,
          typename scalefactor = std::ratio<1, 1>>
static dist_t L2Sqr(const data_t *__restrict pVect1,
                    const data_t *__restrict pVect2, const size_t qty) {
  dist_t res = 0;

  for (size_t i = 0; i < qty / K; i++) {
    for (size_t j = 0; j < K; j++) {
      const size_t index = (i * K) + j;
      dist_t _a = pVect1[index];
      dist_t _b = pVect2[index];
      res += (_a - _b) * (_a - _b);
    }
  }

  constexpr dist_t scale = (dist_t)scalefactor::num / (dist_t)scalefactor::den;
  return (res * scale * scale);
}

template <typename dist_t, typename data_t = dist_t, int K,
          typename scalefactor = std::ratio<1, 1>>
static dist_t L2SqrAtLeast(const data_t *__restrict pVect1,
                           const data_t *__restrict pVect2, const size_t qty) {
  size_t remainder = qty - K;

  return L2Sqr<dist_t, data_t, K, scalefactor>(pVect1, pVect2, K) +
         L2Sqr<dist_t, data_t, 1, scalefactor>(pVect1 + K, pVect2 + K,
                                               remainder);
}



template <SIMD_ARCH s, typename data_t, typename scalefactor = std::ratio<1, 1>>
static float L2SqrSimd(const data_t *__restrict pVect1,
                    const data_t *__restrict pVect2, const size_t qty) {

  typename SimdType<s>::register_t accumulator = SimdType<s>::newAccumulator();

  for (size_t i = 0; i < qty / SimdType<s>::inner_loop_lenght; i++) {

    typename SimdType<s>::register_t v1 = SimdType<s>::loadAndConvertToFloat(pVect1);
    typename SimdType<s>::register_t v2 = SimdType<s>::loadAndConvertToFloat(pVect2);

    if constexpr (s == SSE) {
      L2SqrSSEInnerLoop(v1, v2, accumulator);
    } else if constexpr (s == AVX2) {
      L2SqrAVX2InnerLoop(v1, v2, accumulator);
    } else if constexpr (s == AVX512) {
      L2SqrAVX512FInnerLoop(v1, v2, accumulator);
    }

    pVect1+=SimdType<s>::floatsPerLine;
    pVect2+=SimdType<s>::floatsPerLine;

  }

  float res = SimdType<s>::collapseAccumulator(accumulator);
  constexpr float scale = (float)scalefactor::num / (float)scalefactor::den;
  return (res * scale * scale);
}


static void PORTABLE_TARGET_SSE L2SqrSSEInnerLoop(SimdType<SSE>::register_t v1, SimdType<SSE>::register_t v2, SimdType<SSE>::register_t accumulator) {
    typename SimdType<SSE>::register_t diff = _mm_sub_ps(v1, v2);
    _mm_add_ps(accumulator, _mm_mul_ps(diff, diff));
  }

static void PORTABLE_TARGET_AVX2 L2SqrAVX2InnerLoop(SimdType<AVX2>::register_t v1, SimdType<AVX2>::register_t v2, SimdType<AVX2>::register_t accumulator) {
    typename SimdType<AVX2>::register_t diff = _mm256_sub_ps(v1, v2);
    _mm256_fmadd_ps(diff, diff, accumulator);  
  }

static void PORTABLE_TARGET_AVX512 L2SqrAVX512FInnerLoop(SimdType<AVX512>::register_t v1, SimdType<AVX512>::register_t v2, SimdType<AVX512>::register_t accumulator) {
    typename SimdType<AVX512>::register_t diff = _mm512_sub_ps(v1, v2);
    // sum = _mm512_fmadd_ps(diff, diff, sum);
    _mm512_add_ps(accumulator, _mm512_mul_ps(diff, diff));
  }


template <SIMD_ARCH s, typename data_t, typename scalefactor = std::ratio<1, 1>>
static float L2SqrAtLeastSimd(const float *pVect1, const float *pVect2,
                                     const size_t qty) {
  size_t qty_simd = qty >> SimdType<s>::floatsPerLine << SimdType<s>::floatsPerLine;
  float res = L2SqrSimd<s, data_t, scalefactor>(pVect1, pVect2, qty_simd);

  size_t qty_left = qty - qty_simd;
  float res_tail =
      L2Sqr<float, data_t>(pVect1 + qty_simd, pVect2 + qty_simd, qty_left);
  return (res + res_tail);
}

// #endif

template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
class EuclideanSpace : public Space<dist_t, data_t> {
  DISTFUNC<dist_t, data_t> fstdistfunc_;
  size_t data_size_;
  size_t dim_;

public:
  EuclideanSpace(size_t dim) : data_size_(dim * sizeof(data_t)), dim_(dim) {
    if (dim % 128 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 128, scalefactor>;
    else if (dim % 64 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 64, scalefactor>;
    else if (dim % 32 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 32, scalefactor>;
    else if (dim % 16 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 16, scalefactor>;
    else if (dim % 8 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 8, scalefactor>;
    else if (dim % 4 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 4, scalefactor>;

    else if (dim > 128)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 128, scalefactor>;
    else if (dim > 64)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 64, scalefactor>;
    else if (dim > 32)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 32, scalefactor>;
    else if (dim > 16)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 16, scalefactor>;
    else if (dim > 8)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 8, scalefactor>;
    else if (dim > 4)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 4, scalefactor>;
    else
      fstdistfunc_ = L2Sqr<dist_t, data_t, 1, scalefactor>;
  }

  size_t get_data_size() { return data_size_; }

  DISTFUNC<dist_t, data_t> get_dist_func() { return fstdistfunc_; }

  size_t get_dist_func_param() { return dim_; }

  ~EuclideanSpace() {}

  template <typename T = dist_t, typename U = data_t,
              typename std::enable_if<std::is_same<T, float>::value && std::is_same<U, float>::value, int>::type = 0>
  EuclideanSpace(size_t dim)
      : data_size_(dim * sizeof(float)), dim_(dim) {
    fstdistfunc_ = L2Sqr<float, float>;
    SIMD_ARCH simd_arch = get_x86_simd_arch();

    if (dim % 128 == 0)
      fstdistfunc_ = L2SqrSimd<SSE, float, 128, scalefactor>;
    else if (dim % 64 == 0)
      fstdistfunc_ = L2SqrSimd<SSE, float, 64, scalefactor>;
    else if (dim % 32 == 0)
      fstdistfunc_ = L2SqrSimd<SSE, float, 32, scalefactor>;
    else if (dim % 16 == 0)
      fstdistfunc_ = L2SqrSimd<SSE, float, 16, scalefactor>;
    else if (dim % 8 == 0)
      fstdistfunc_ = L2SqrSimd<SSE, float, 8, scalefactor>;
    else if (dim % 4 == 0)
      fstdistfunc_ = L2SqrSimd<SSE, float, 4, scalefactor>;

    else if (dim > 128)
      fstdistfunc_ = L2SqrAtLeastSimd<SSE, float, 128, scalefactor>;
    else if (dim > 64)
      fstdistfunc_ = L2SqrAtLeastSimd<SSE, float, 64, scalefactor>;
    else if (dim > 32)
      fstdistfunc_ = L2SqrAtLeastSimd<SSE, float, 32, scalefactor>;
    else if (dim > 16)
      fstdistfunc_ = L2SqrAtLeastSimd<SSE, float, 16, scalefactor>;
    else if (dim > 8)
      fstdistfunc_ = L2SqrAtLeastSimd<SSE, float, 8, scalefactor>;
    else if (dim > 4)
      fstdistfunc_ = L2SqrAtLeastSimd<SSE, float, 4, scalefactor>;
    else
      fstdistfunc_ = L2Sqr<float, float, 1, scalefactor>;
    }

};


} // namespace hnswlib
