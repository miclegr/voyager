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
#include "E4M3.h"
#include <iostream>

#if defined(__GNUC__)
#define PORTABLE_TARGET_SSE __attribute__((target("sse4.1")))
#define PORTABLE_TARGET_AVX2 __attribute__((target("avx,avx2,fma")))
#define PORTABLE_TARGET_AVX512 __attribute__((target("avx512f,avx512dq")))
#else
#define PORPORTABLE_TARGET_SSE
#define PORTABLE_TARGET_AVX2
#define PORPORTABLE_TARGET_AVX512
#endif

enum SIMD_ARCH {
  NONE,
  SSE,
  AVX2,
  AVX512,
};

SIMD_ARCH get_x86_simd_arch(){
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


template<SIMD_ARCH s>
struct SimdType{};

template<>
struct SimdType<SSE>{
  static constexpr int floatsPerLine = 4;
  using register_t = __m128;

  template<typename data_t>
  static register_t loadAndConvertToFloat(data_t *data);

  static register_t newAccumulator();
  
  static float collapseAccumulator(register_t& accumulator);
};

template<>
struct SimdType<AVX2>{
  static constexpr int floatsPerLine = 8;
  using register_t = __m256;

  template<typename data_t>
  static register_t loadAndConvertToFloat(data_t *data);

  static register_t newAccumulator();
  
  static float collapseAccumulator(register_t& accumulator);
};

template<>
struct SimdType<AVX512>{
  static constexpr int floatsPerLine = 16;
  using register_t = __m512;

  template<typename data_t>
  static register_t loadAndConvertToFloat(data_t *data);
  
  static register_t newAccumulator();
  
  static float collapseAccumulator(register_t& accumulator);
};



template<>
typename SimdType<SSE>::register_t PORTABLE_TARGET_SSE SimdType<SSE>::loadAndConvertToFloat(float* data){
    return _mm_loadu_ps(data);
}

template<>
typename SimdType<AVX2>::register_t PORTABLE_TARGET_AVX2 SimdType<AVX2>::loadAndConvertToFloat(float* data){
    return _mm256_loadu_ps(data);
}

template<>
typename SimdType<AVX512>::register_t PORTABLE_TARGET_AVX512 SimdType<AVX512>::loadAndConvertToFloat(float* data){
    return _mm512_loadu_ps(data);
}

template<>
typename SimdType<SSE>::register_t PORTABLE_TARGET_SSE SimdType<SSE>::loadAndConvertToFloat(int8_t* data){
    __m128i int_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
    return _mm_cvtepi32_ps(_mm_cvtepi8_epi32(int_vector));
}

template<>
typename SimdType<AVX2>::register_t PORTABLE_TARGET_AVX2 SimdType<AVX2>::loadAndConvertToFloat(int8_t* data){
    __m128i int_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(int_vector));
}

template<>
typename SimdType<AVX512>::register_t PORTABLE_TARGET_AVX512 SimdType<AVX512>::loadAndConvertToFloat(int8_t* data){
    __m128i int_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(int_vector));
}

// template<>
// typename SimdType<SSE>::vector convert_to_float_vector<SSE>(E4M3* data){
// 
//     __m128i e4m3_vector = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(data));
//     
//     // Check if e4m3_vector is all zeros or all ones
//     __m128i is_subnormal = _mm_cmpeq_epi8(e4m3_vector,_mm_setzero_si128());
//     __m128i is_nan = _mm_cmpeq_epi8(e4m3_vector, _mm_set1_epi8(0xFF));
//     __m128i is_subnormal_or_nan = _mm_cvtepu8_epi32(
//         _mm_or_si128(is_subnormal, is_nan)
//     );
// 
//     // subnormal or nan case
//     __m128i e4m3_sign_masked = _mm_and_si128(e4m3_vector, _mm_set1_epi8(0x7F));
// 
//     const __m128i subnormal_exponent_lookup = _mm_set_epi8(0xFF, 0x00, 0x00,0x00 0x00,0x00,0x00,0x00,0x77,0x77,0x77,0x77,0x76,0x76,0x75,0x00);
//     const __m128i subnormal_mantissa_lookup = _mm_set_epi8(0x01, 0x00, 0x00,0x00 0x00,0x00,0x00,0x00,0x06,0x04,0x02,0x00,0x04,0x00,0x00,0x00);
// 
//     __m128i subnormal_exponent_mapped = _mm_cvtepu8_epi32(_mm_shuffle_epi8(subnormal_exponent_lookup, e4m3_sign_masked));
//     __m128i subnormal_mantissa_mapped = _mm_cvtepu8_epi32(_mm_shuffle_epi8(subnormal_mantissa_lookup, e4m3_sign_masked));
// 
//     __m128i unsigned_subnormal = _mm_or_si128(
//         _mm_slli_epi32(subnormal_exponent_mapped, 23),
//         _mm_slli_epi32(subnormal_mantissa_mapped, 20)
//     );
// 
//     // normal case
//     __m128i normal_exponent = _mm_cvtepu8_epi32(_mm_and_si128(e4m3_vector,_mm_set1_epi8(0x78)));
//     __m128i normal_mantissa = _mm_cvtepu8_epi32(_mm_and_si128(e4m3_vector,_mm_set1_epi8(0x07)));
//     normal_exponent_ = _mm_add_epi32(normal_exponent, __mm_set1_epi32(120));
// 
//     __m128i unsigned_normal = _mm_or_si128(
//         _mm_slli_epi32(normal_exponent, 23),
//         _mm_slli_epi32(normal_mantissa, 20)
//     );
// 
//     // combine
//     __m128i blended = _mm_blendv_ps(is_subnormal_or_nan, usigned_subnormal, unsigned_normal);
// 
//     const __m128i sign = _mm_cvtepu8_epi32(_mm_and_si128(e4m3_vector,_mm_set1_epi8(0x80)));
//     return _mm_or_si128(blended, sign);
//     
// }


typename SimdType<SSE>::register_t PORTABLE_TARGET_SSE SimdType<SSE>::newAccumulator(){
  return _mm_set1_ps(0);
}

typename SimdType<AVX2>::register_t PORTABLE_TARGET_AVX2 SimdType<AVX2>::newAccumulator(){
  return _mm256_set1_ps(0);
}

typename SimdType<AVX512>::register_t PORTABLE_TARGET_AVX512 SimdType<AVX512>::newAccumulator(){
  return _mm512_set1_ps(0);
}


float PORTABLE_TARGET_SSE SimdType<SSE>::collapseAccumulator(typename SimdType<SSE>::register_t& vec) {
    __m128 shuf = _mm_movehdup_ps(vec);
    __m128 sums = _mm_add_ps(vec, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

float PORTABLE_TARGET_AVX2 SimdType<AVX2>::collapseAccumulator(typename SimdType<AVX2>::register_t& vec) {
    __m256 shuf = _mm256_movehdup_ps(vec);
    __m256 sums = _mm256_add_ps(vec, shuf);
    shuf = _mm256_permute2f128_ps(sums, sums, 0x31);
    sums = _mm256_add_ps(sums, shuf);
    __m128 low = _mm256_castps256_ps128(sums);
    __m128 high = _mm256_extractf128_ps(sums, 1);
    low = _mm_add_ps(low, high);
    low = _mm_hadd_ps(low, low);
    low = _mm_hadd_ps(low, low);
    return _mm_cvtss_f32(low);
}

float PORTABLE_TARGET_AVX512 SimdType<AVX512>::collapseAccumulator(typename SimdType<AVX512>::register_t& vec) {
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
