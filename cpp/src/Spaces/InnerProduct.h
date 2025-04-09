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
#include <ratio>

namespace hnswlib {
/**
 * For a given loop unrolling factor K, distance type dist_t, and data type
 * data_t, calculate the inner product distance between two vectors. The
 * compiler should automatically do the loop unrolling for us here and vectorize
 * as appropriate.
 */
template <typename dist_t, typename data_t = dist_t, int K = 1,
          typename scalefactor = std::ratio<1, 1>>
static dist_t InnerProductWithoutScale(const data_t *pVect1,
                                       const data_t *pVect2, size_t qty) {
  dist_t res = 0;

  qty = qty / K;

  for (size_t i = 0; i < qty; i++) {
    for (size_t j = 0; j < K; j++) {
      const size_t index = (i * K) + j;
      dist_t _a = pVect1[index];
      dist_t _b = pVect2[index];
      res += _a * _b;
    }
  }
  return res;
}

template <typename dist_t, typename scalefactor = std::ratio<1, 1>>
static dist_t ScaleInnerProduct(dist_t res) {
  constexpr dist_t scale = (dist_t)scalefactor::num / (dist_t)scalefactor::den;
  res *= scale * scale;
  res = (static_cast<dist_t>(1.0f) - res);
  return res;
}

template <typename dist_t, typename data_t = dist_t, int K = 1,
          typename scalefactor = std::ratio<1, 1>>
static dist_t InnerProduct(const data_t *pVect1, const data_t *pVect2,
                           size_t qty) {
  dist_t res = InnerProductWithoutScale<dist_t, data_t, K, scalefactor>(
      pVect1, pVect2, qty);
  return ScaleInnerProduct<dist_t, scalefactor>(res);
}

template <typename dist_t, typename data_t = dist_t, int K,
          typename scalefactor = std::ratio<1, 1>>
static dist_t InnerProductAtLeast(const data_t *__restrict pVect1,
                                  const data_t *__restrict pVect2,
                                  const size_t qty) {
  size_t remainder = qty - K;
  dist_t res = InnerProductWithoutScale<dist_t, data_t, K, scalefactor>(
                   pVect1, pVect2, K) +
               InnerProductWithoutScale<dist_t, data_t, 1, scalefactor>(
                   pVect1 + K, pVect2 + K, remainder);
  return ScaleInnerProduct<dist_t, scalefactor>(res);
}

template <SIMD_ARCH arch, typename data_t>
static float InnerProductWithoutScaleSimd(const data_t *__restrict pVect1,
                    const data_t *__restrict pVect2, const size_t qty) {

  using Simd = SimdType<arch>;
  using register_t = typename Simd::register_t;
  float res;

  if constexpr (arch == SIMD_ARCH::SSE) {
    res = [&]() PORTABLE_TARGET_SSE {

      register_t accumulator = Simd::newAccumulator();
      for (size_t i = 0; i < qty / Simd::floatsPerLine; i++) {

        register_t v1 = Simd::loadAndConvertToFloat(pVect1);
        register_t v2 = Simd::loadAndConvertToFloat(pVect2);
        accumulator = _mm_add_ps(accumulator, _mm_mul_ps(v1, v2));
        pVect1+=Simd::floatsPerLine;
        pVect2+=Simd::floatsPerLine;

      }
      float res = Simd::collapseAccumulator(accumulator);
      return res;
      }();
  } else if constexpr (arch == SIMD_ARCH::AVX2) {
    res = [&]() PORTABLE_TARGET_AVX2 {

      register_t accumulator = Simd::newAccumulator();
      for (size_t i = 0; i < qty / Simd::floatsPerLine; i++) {

        register_t v1 = Simd::loadAndConvertToFloat(pVect1);
        register_t v2 = Simd::loadAndConvertToFloat(pVect2);
        accumulator = _mm256_fmadd_ps(v1, v2, accumulator);  
        pVect1+=Simd::floatsPerLine;
        pVect2+=Simd::floatsPerLine;

      }
      float res = Simd::collapseAccumulator(accumulator);
      return res;
      }();
  } else if constexpr (arch == SIMD_ARCH::AVX512) {
    res = [&]() PORTABLE_TARGET_AVX512 {

      register_t accumulator = Simd::newAccumulator();
      for (size_t i = 0; i < qty / Simd::floatsPerLine; i++) {

        register_t v1 = Simd::loadAndConvertToFloat(pVect1);
        register_t v2 = Simd::loadAndConvertToFloat(pVect2);
        accumulator = _mm512_fmadd_ps(v1, v2, accumulator);
        pVect1+=Simd::floatsPerLine;
        pVect2+=Simd::floatsPerLine;

      }
      float res = Simd::collapseAccumulator(accumulator);
      return res;
      }();
  }

  return res;
}

template <SIMD_ARCH arch, typename data_t, typename scalefactor = std::ratio<1, 1>>
static float InnerProductSimd(const data_t *pVect1, const data_t *pVect2,
                           size_t qty) {
  float res = InnerProductWithoutScaleSimd<arch, data_t>(
      pVect1, pVect2, qty);
  return ScaleInnerProduct<float, scalefactor>(res);
}


template <SIMD_ARCH arch, typename data_t, typename scalefactor = std::ratio<1, 1>>
static float InnerProductAtLeastSimd(const data_t *pVect1, const data_t *pVect2,
                                     const size_t qty) {
  size_t qty_simd = qty >> SimdType<arch>::floatsPerLine << SimdType<arch>::floatsPerLine;
  float res = InnerProductWithoutScaleSimd<arch, data_t>(pVect1, pVect2, qty_simd);

  size_t qty_left = qty - qty_simd;
  float res_tail =
      InnerProductWithoutScale<float, data_t, 1, scalefactor>(pVect1 + qty_simd, pVect2 + qty_simd, qty_left);
  return ScaleInnerProduct<float, scalefactor>(res + res_tail);
}


template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
class InnerProductSpace : public Space<dist_t, data_t> {
  DISTFUNC<dist_t, data_t> fstdistfunc_;
  size_t data_size_;
  size_t dim_;

public:
  InnerProductSpace(size_t dim) : data_size_(dim * sizeof(data_t)), dim_(dim) {
    if (dim % 128 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 128, scalefactor>;
    else if (dim % 64 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 64, scalefactor>;
    else if (dim % 32 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 32, scalefactor>;
    else if (dim % 16 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 16, scalefactor>;
    else if (dim % 8 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 8, scalefactor>;
    else if (dim % 4 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 4, scalefactor>;

    else if (dim > 128)
      fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 128, scalefactor>;
    else if (dim > 64)
      fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 64, scalefactor>;
    else if (dim > 32)
      fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 32, scalefactor>;
    else if (dim > 16)
      fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 16, scalefactor>;
    else if (dim > 8)
      fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 8, scalefactor>;
    else if (dim > 4)
      fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 4, scalefactor>;
    else
      fstdistfunc_ = InnerProduct<dist_t, data_t, 1, scalefactor>;

    if constexpr (std::is_same<dist_t, float>::value) {

      SIMD_ARCH simd_arch = getX86SimdArch();

      if (simd_arch == SIMD_ARCH::SSE and dim % SimdType<SIMD_ARCH::SSE>::floatsPerLine == 0)
        fstdistfunc_ = InnerProductSimd<SSE, data_t, scalefactor>;
      else if (simd_arch == SIMD_ARCH::SSE and dim > SimdType<SIMD_ARCH::SSE>::floatsPerLine)
        fstdistfunc_ = InnerProductAtLeastSimd<SSE, data_t, scalefactor>;
      else if (simd_arch == SIMD_ARCH::AVX2 and dim % SimdType<SIMD_ARCH::AVX2>::floatsPerLine == 0)
        fstdistfunc_ = InnerProductSimd<AVX2, data_t, scalefactor>;
      else if (simd_arch == SIMD_ARCH::AVX2 and dim > SimdType<SIMD_ARCH::AVX2>::floatsPerLine)
        fstdistfunc_ = InnerProductAtLeastSimd<AVX2, data_t, scalefactor>;
      else if (simd_arch == SIMD_ARCH::AVX512 and dim % SimdType<SIMD_ARCH::AVX512>::floatsPerLine == 0)
        fstdistfunc_ = InnerProductSimd<AVX512, data_t, scalefactor>;
      else if (simd_arch == SIMD_ARCH::AVX512 and dim > SimdType<SIMD_ARCH::AVX512>::floatsPerLine)
        fstdistfunc_ = InnerProductAtLeastSimd<AVX512, data_t, scalefactor>;

      }
  }

  size_t get_data_size() { return data_size_; }

  DISTFUNC<dist_t, data_t> get_dist_func() { return fstdistfunc_; }

  size_t get_dist_func_param() { return dim_; }
  ~InnerProductSpace() {}
};

} // namespace hnswlib
