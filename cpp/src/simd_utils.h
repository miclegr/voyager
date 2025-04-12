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

#include <iostream>
#include <memory>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_TARGET_SSE __attribute__((target("sse4.2")))
#define PORTABLE_TARGET_AVX2 __attribute__((target("avx,avx2,fma")))
#define PORTABLE_TARGET_AVX512 __attribute__((target("avx512f,avx512dq")))
#else
#define PORTABLE_TARGET_SSE
#define PORTABLE_TARGET_AVX2
#define PORTABLE_TARGET_AVX512
#endif

#include "E4M3.h"

enum SIMD_ARCH {
  NONE,
  SSE,
  AVX2,
  AVX512,
};

SIMD_ARCH getX86SimdArch() {
#ifdef _WIN32
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);
  if (cpuInfo[0] >= 7) {
    __cpuid(cpuInfo, 7);
    if (cpuInfo[1] & (1 << 16)) {
      return AVX512;
    }
  }
  __cpuid(cpuInfo, 1);
  if (cpuInfo[2] & (1 << 28)) {
    return AVX2;
  }
  return SSE;
#else
  if (__builtin_cpu_supports("avx512f")) {
    return AVX512;
  } else if (__builtin_cpu_supports("avx")) {
    return AVX2;
  } else if (__builtin_cpu_supports("sse4.2")) {
    return SSE;
  }
  return NONE;
#endif
}

template <SIMD_ARCH s> struct SimdType {};

template <> struct SimdType<SSE> {
  static constexpr int floatsPerLine = 4;
  using register_t = __m128;

  template <typename data_t>
  static register_t loadAndConvertToFloat(data_t const *data);

  static register_t newAccumulator();

  static float collapseAccumulator(register_t &accumulator);

  template <typename Func, typename... Args>
  static std::vector<float> registerToVectorWrapper(Func &&func,
                                                    Args &&...args);
};

template <> struct SimdType<AVX2> {
  static constexpr int floatsPerLine = 8;
  using register_t = __m256;

  template <typename data_t>
  static register_t loadAndConvertToFloat(data_t const *data);

  static register_t newAccumulator();

  static float collapseAccumulator(register_t &accumulator);

  template <typename Func, typename... Args>
  static std::vector<float> registerToVectorWrapper(Func &&func,
                                                    Args &&...args);
};

template <> struct SimdType<AVX512> {
  static constexpr int floatsPerLine = 16;
  using register_t = __m512;

  template <typename data_t>
  static register_t loadAndConvertToFloat(data_t const *data);

  static register_t newAccumulator();

  static float collapseAccumulator(register_t &accumulator);

  template <typename Func, typename... Args>
  static std::vector<float> registerToVectorWrapper(Func &&func,
                                                    Args &&...args);
};

template <>
typename SimdType<SSE>::register_t PORTABLE_TARGET_SSE
SimdType<SSE>::loadAndConvertToFloat(float const *data) {
  return _mm_loadu_ps(data);
}

template <>
typename SimdType<AVX2>::register_t PORTABLE_TARGET_AVX2
SimdType<AVX2>::loadAndConvertToFloat(float const *data) {
  return _mm256_loadu_ps(data);
}

template <>
typename SimdType<AVX512>::register_t PORTABLE_TARGET_AVX512
SimdType<AVX512>::loadAndConvertToFloat(float const *data) {
  return _mm512_loadu_ps(data);
}

template <>
typename SimdType<SSE>::register_t PORTABLE_TARGET_SSE
SimdType<SSE>::loadAndConvertToFloat(int8_t const *data) {
  __m128i int_vector = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(data));
  return _mm_cvtepi32_ps(_mm_cvtepi8_epi32(int_vector));
}

template <>
typename SimdType<AVX2>::register_t PORTABLE_TARGET_AVX2
SimdType<AVX2>::loadAndConvertToFloat(int8_t const *data) {
  __m128i int_vector =
      _mm_cvtsi64_si128(*reinterpret_cast<const int64_t *>(data));
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(int_vector));
}

template <>
typename SimdType<AVX512>::register_t PORTABLE_TARGET_AVX512
SimdType<AVX512>::loadAndConvertToFloat(int8_t const *data) {
  __m128i int_vector = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(data));
  return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(int_vector));
}

template <>
typename SimdType<SSE>::register_t PORTABLE_TARGET_SSE
SimdType<SSE>::loadAndConvertToFloat(E4M3 const *data) {

  __m128i e4m3_vector = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(data));
  __m128i e4m3_vector_32 = _mm_cvtepu8_epi32(e4m3_vector);

  // Check if e4m3_vector is all zeros or all ones
  __m128i is_subnormal =
      _mm_cmpeq_epi32(_mm_and_si128(e4m3_vector_32, _mm_set1_epi32(0b00011110)),
                      _mm_setzero_si128());
  __m128i is_nan =
      _mm_cmpeq_epi32(_mm_and_si128(e4m3_vector_32, _mm_set1_epi32(0b11111110)),
                      _mm_set1_epi32(0b11111110));
  __m128 is_subnormal_or_nan =
      _mm_castsi128_ps(_mm_or_si128(is_subnormal, is_nan));

  // subnormal or nan case

  const __m128i subnormal_exponent_lookup =
      _mm_set_epi8(0xFF, 0x77, 0x00, 0x77, 0x00, 0x77, 0x00, 0x77, 0x00, 0x76,
                   0x00, 0x76, 0x00, 0x75, 0x00, 0x00);
  const __m128i subnormal_mantissa_lookup =
      _mm_set_epi8(0x01, 0x06, 0x00, 0x04, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04,
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

  __m128i subnormal_lookup_idx = _mm_srli_epi64(e4m3_vector, 4);
  __m128i subnormal_exponent_mapped = _mm_cvtepu8_epi32(
      _mm_shuffle_epi8(subnormal_exponent_lookup, subnormal_lookup_idx));
  __m128i subnormal_mantissa_mapped = _mm_cvtepu8_epi32(
      _mm_shuffle_epi8(subnormal_mantissa_lookup, subnormal_lookup_idx));

  __m128 unsigned_subnormal = _mm_castsi128_ps(
      _mm_or_si128(_mm_slli_epi32(subnormal_exponent_mapped, 23),
                   _mm_slli_epi32(subnormal_mantissa_mapped, 20)));

  // normal case
  __m128i normal_exponent =
      _mm_and_si128(e4m3_vector_32, _mm_set1_epi32(0b00011110));
  __m128i normal_mantissa =
      _mm_and_si128(e4m3_vector_32, _mm_set1_epi32(0b11100000));
  normal_exponent =
      _mm_add_epi32(normal_exponent, _mm_set1_epi32(240)); // 240 = (120 << 1)

  __m128 unsigned_normal =
      _mm_castsi128_ps(_mm_or_si128(_mm_slli_epi32(normal_exponent, 22),
                                    _mm_slli_epi32(normal_mantissa, 15)));

  // combine
  __m128i blended = _mm_castps_si128(
      _mm_blendv_ps(unsigned_normal, unsigned_subnormal, is_subnormal_or_nan));

  const __m128i sign =
      _mm_slli_epi32(_mm_and_si128(e4m3_vector_32, _mm_set1_epi32(1)), 31);
  return _mm_castsi128_ps(_mm_or_si128(blended, sign));
}

template <>
typename SimdType<AVX2>::register_t PORTABLE_TARGET_AVX2
SimdType<AVX2>::loadAndConvertToFloat(E4M3 const *data) {

  __m128i e4m3_vector =
      _mm_cvtsi64_si128(*reinterpret_cast<const int64_t *>(data));
  __m256i e4m3_vector_32 = _mm256_cvtepu8_epi32(e4m3_vector);

  const __m256i exponent_mask = _mm256_set1_epi32(0b00011110);
  const __m256i mantissa_mask = _mm256_set1_epi32(0b11100000);

  // subnormal or nan case

  const __m256i subnormal_exponent_lookup = _mm256_castsi128_si256(
      _mm_set_epi8(0xFF, 0x77, 0x00, 0x77, 0x00, 0x77, 0x00, 0x77, 0x00, 0x76,
                   0x00, 0x76, 0x00, 0x75, 0x00, 0x00));
  const __m256i subnormal_mantissa_lookup = _mm256_castsi128_si256(
      _mm_set_epi8(0x01, 0x06, 0x00, 0x04, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04,
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00));

  __m256i subnormal_lookup_idx =
      _mm256_srli_epi64(_mm256_castsi128_si256(e4m3_vector), 4);
  __m256i subnormal_exponent_mapped =
      _mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_shuffle_epi8(
          subnormal_exponent_lookup, subnormal_lookup_idx)));
  __m256i subnormal_mantissa_mapped =
      _mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_shuffle_epi8(
          subnormal_mantissa_lookup, subnormal_lookup_idx)));

  __m256 unsigned_subnormal = _mm256_castsi256_ps(
      _mm256_or_si256(_mm256_slli_epi32(subnormal_exponent_mapped, 23),
                      _mm256_slli_epi32(subnormal_mantissa_mapped, 20)));

  // normal case
  const __m256i normal_exponent_bias =
      _mm256_set1_epi32(240); // 240 = (120 << 1)

  __m256i normal_exponent = _mm256_and_si256(e4m3_vector_32, exponent_mask);
  __m256i normal_mantissa = _mm256_and_si256(e4m3_vector_32, mantissa_mask);
  normal_exponent = _mm256_add_epi32(normal_exponent, normal_exponent_bias);

  __m256 unsigned_normal = _mm256_castsi256_ps(
      _mm256_or_si256(_mm256_slli_epi32(normal_exponent, 22),
                      _mm256_slli_epi32(normal_mantissa, 15)));

  // combine

  // Check if e4m3_vector is all zeros or all ones
  __m256i is_subnormal = _mm256_cmpeq_epi32(
      _mm256_and_si256(e4m3_vector_32, exponent_mask), _mm256_setzero_si256());

  //__m256i nan_pattern =
  //    _mm256_srli_epi32(_mm256_or_si256(exponent_mask, mantissa_mask), 1);
  //__m256i is_nan =
  //    _mm256_cmpeq_epi32(_mm256_srli_epi32(e4m3_vector_32, 1), nan_pattern);

  __m256 is_subnormal_or_nan = _mm256_castsi256_ps(is_subnormal);
  //    _mm256_castsi256_ps(_mm256_or_si256(is_subnormal, is_nan));

  __m256i blended = _mm256_castps_si256(_mm256_blendv_ps(
      unsigned_normal, unsigned_subnormal, is_subnormal_or_nan));

  const __m256i sign = _mm256_slli_epi32(e4m3_vector_32, 31);
  return _mm256_castsi256_ps(_mm256_or_si256(blended, sign));
}

template <>
typename SimdType<AVX512>::register_t PORTABLE_TARGET_AVX512
SimdType<AVX512>::loadAndConvertToFloat(E4M3 const *data) {

  __m128i e4m3_vector =
      _mm_lddqu_si128(reinterpret_cast<const __m128i *>(data));
  __m512i e4m3_vector_32 = _mm512_cvtepu8_epi32(e4m3_vector);

  // Check if e4m3_vector is all zeros or all ones
  __mmask16 is_subnormal = _mm512_cmpeq_epi32_mask(
      _mm512_and_si512(e4m3_vector_32, _mm512_set1_epi32(0b00011110)),
      _mm512_setzero_si512());
  __mmask16 is_nan = _mm512_cmpeq_epi32_mask(
      _mm512_and_si512(e4m3_vector_32, _mm512_set1_epi32(0b11111110)),
      _mm512_set1_epi32(0b11111110));
  __mmask16 is_subnormal_or_nan = is_subnormal | is_nan;

  // subnormal or nan case

  const __m256i subnormal_exponent_lookup = _mm256_castsi128_si256(
      _mm_set_epi8(0xFF, 0x77, 0x00, 0x77, 0x00, 0x77, 0x00, 0x77, 0x00, 0x76,
                   0x00, 0x76, 0x00, 0x75, 0x00, 0x00));
  const __m256i subnormal_mantissa_lookup = _mm256_castsi128_si256(
      _mm_set_epi8(0x01, 0x06, 0x00, 0x04, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04,
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00));

  __m256i subnormal_lookup_idx =
      _mm256_srli_epi64(_mm256_castsi128_si256(e4m3_vector), 4);
  __m512i subnormal_exponent_mapped =
      _mm512_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_shuffle_epi8(
          subnormal_exponent_lookup, subnormal_lookup_idx)));
  __m512i subnormal_mantissa_mapped =
      _mm512_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_shuffle_epi8(
          subnormal_mantissa_lookup, subnormal_lookup_idx)));

  __m512 unsigned_subnormal = _mm512_castsi512_ps(
      _mm512_or_si512(_mm512_slli_epi32(subnormal_exponent_mapped, 23),
                      _mm512_slli_epi32(subnormal_mantissa_mapped, 20)));

  // normal case
  __m512i normal_exponent =
      _mm512_and_si512(e4m3_vector_32, _mm512_set1_epi32(0b00011110));
  __m512i normal_mantissa =
      _mm512_and_si512(e4m3_vector_32, _mm512_set1_epi32(0b11100000));
  normal_exponent = _mm512_add_epi32(
      normal_exponent, _mm512_set1_epi32(240)); // 240 = (120 << 1)

  __m512 unsigned_normal = _mm512_castsi512_ps(
      _mm512_or_si512(_mm512_slli_epi32(normal_exponent, 22),
                      _mm512_slli_epi32(normal_mantissa, 15)));

  // combine
  __m512i blended = _mm512_castps_si512(_mm512_mask_blend_ps(
      is_subnormal_or_nan, unsigned_normal, unsigned_subnormal));

  const __m512i sign = _mm512_slli_epi32(
      _mm512_and_si512(e4m3_vector_32, _mm512_set1_epi32(1)), 31);
  return _mm512_castsi512_ps(_mm512_or_si512(blended, sign));
}

typename SimdType<SSE>::register_t PORTABLE_TARGET_SSE
SimdType<SSE>::newAccumulator() {
  return _mm_setzero_ps();
}

typename SimdType<AVX2>::register_t PORTABLE_TARGET_AVX2
SimdType<AVX2>::newAccumulator() {
  return _mm256_setzero_ps();
}

typename SimdType<AVX512>::register_t PORTABLE_TARGET_AVX512
SimdType<AVX512>::newAccumulator() {
  return _mm512_setzero_ps();
}

float PORTABLE_TARGET_SSE
SimdType<SSE>::collapseAccumulator(typename SimdType<SSE>::register_t &vec) {
  __m128 shuf = _mm_movehdup_ps(vec);
  __m128 sums = _mm_add_ps(vec, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

float PORTABLE_TARGET_AVX2
SimdType<AVX2>::collapseAccumulator(typename SimdType<AVX2>::register_t &vec) {
  __m128 hiQuad = _mm256_extractf128_ps(vec, 1);
  __m128 loQuad = _mm256_castps256_ps128(vec);
  __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  __m128 loDual = sumQuad;
  __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  __m128 sumDual = _mm_add_ps(loDual, hiDual);
  __m128 lo = sumDual;
  __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  __m128 sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}

float PORTABLE_TARGET_AVX512 SimdType<AVX512>::collapseAccumulator(
    typename SimdType<AVX512>::register_t &vec) {
  __m256 low = _mm512_castps512_ps256(vec);
  __m256 high = _mm512_extractf32x8_ps(vec, 1);
  low = _mm256_add_ps(low, high);
  __m128 low128 = _mm256_castps256_ps128(low);
  __m128 high128 = _mm256_extractf128_ps(low, 1);
  low128 = _mm_add_ps(low128, high128);
  low128 = _mm_hadd_ps(low128, low128);
  low128 = _mm_hadd_ps(low128, low128);
  return _mm_cvtss_f32(low128);
}

template <typename Func, typename... Args>
std::vector<float> PORTABLE_TARGET_SSE
SimdType<SSE>::registerToVectorWrapper(Func &&func, Args &&...args) {
  using Simd = SimdType<SSE>;
  typename Simd::register_t result = func(std::forward<Args>(args)...);
  std::vector<float> float_array(Simd::floatsPerLine);
  std::memcpy(float_array.data(), &result, sizeof(float) * Simd::floatsPerLine);
  return float_array;
}

template <typename Func, typename... Args>
std::vector<float> PORTABLE_TARGET_AVX2
SimdType<AVX2>::registerToVectorWrapper(Func &&func, Args &&...args) {
  using Simd = SimdType<AVX2>;
  typename Simd::register_t result = func(std::forward<Args>(args)...);
  std::vector<float> float_array(Simd::floatsPerLine);
  std::memcpy(float_array.data(), &result, sizeof(float) * Simd::floatsPerLine);
  return float_array;
}

template <typename Func, typename... Args>
std::vector<float> PORTABLE_TARGET_AVX512
SimdType<AVX512>::registerToVectorWrapper(Func &&func, Args &&...args) {
  using Simd = SimdType<AVX512>;
  typename Simd::register_t result = func(std::forward<Args>(args)...);
  std::vector<float> float_array(Simd::floatsPerLine);
  std::memcpy(float_array.data(), &result, sizeof(float) * Simd::floatsPerLine);
  return float_array;
}
