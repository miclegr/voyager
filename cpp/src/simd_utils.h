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


enum SIMD_ARCH {
  SSE,
  AVX2,
  AVX512,
};

template<SIMD_ARCH s>
struct SimdType{};
template<>
struct SimdType<SSE>{
  static constexpr int inner_loop_lenght = 4;
  using vector = __m128;
};
template<>
struct SimdType<AVX2>{
  static constexpr int inner_loop_lenght = 8;
  using vector = __m256;
};
template<>
struct SimdType<AVX512>{
  static constexpr int inner_loop_lenght = 16;
  using vector = __m512;
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
    } else {
        return SSE;
    }
#endif
}


template<SIMD_ARCH s, typename data_t>
typename SimdType<s>::vector convert_to_float_vector(data_t *data);

template<>
typename SimdType<SSE>::vector convert_to_float_vector<SSE>(float* data){
    return _mm_loadu_ps(data);
}

template<>
typename SimdType<AVX2>::vector convert_to_float_vector<AVX2>(float* data){
    return _mm256_loadu_ps(data);
}

template<>
typename SimdType<AVX512>::vector convert_to_float_vector<AVX512>(float* data){
    return _mm512_loadu_ps(data);
}

template<>
typename SimdType<SSE>::vector convert_to_float_vector<SSE>(int8_t* data){
    __m128i int_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
    return _mm_cvtepi32_ps(_mm_cvtepi8_epi32(int_vector));
}

template<>
typename SimdType<AVX2>::vector convert_to_float_vector<AVX2>(int8_t* data){
    __m256i int_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(int_vector));
}

template<>
typename SimdType<AVX512>::vector convert_to_float_vector<AVX512>(int8_t* data){
    __m512i int_vector = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data));
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(int_vector));
}


template<SIMD_ARCH s>
typename SimdType<s>::vector new_accumulator();

template<>
typename SimdType<SSE>::vector new_accumulator<SSE>(){
  return _mm128_set1_ps(0);
}

template<>
typename SimdType<AVX2>::vector new_accumulator<AVX2>(){
  return _mm256_set1_ps(0);
}

template<>
typename SimdType<AVX512>::vector new_accumulator<AVX2>(){
  return _mm512_set1_ps(0);
}


template<SIMD_ARCH s>
float collapse_accumulator(typename SimdType<s>::vector vec);

template<>
float collapse_accumulator<SSE>(typename SimdType<SSE>::vector vec) {
    __m128 shuf = _mm_movehdup_ps(vec);
    __m128 sums = _mm_add_ps(vec, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

template<>
float collapse_accumulator<AVX2>(typename SimdType<AVX2>::vector vec) {
    __m256 shuf = _mm256_movehdup_ps(vec);
    __m256 sums = _mm256_add_ps(vec, shuf);
    shuf = _mm256_movehl_ps(shuf, sums);
    sums = _mm256_add_ps(sums, shuf);
    __m128 low = _mm256_castps256_ps128(sums);
    __m128 high = _mm256_extractf128_ps(sums, 1);
    low = _mm_add_ps(low, high);
    low = _mm_hadd_ps(low, low);
    low = _mm_hadd_ps(low, low);
    return _mm_cvtss_f32(low);
}

template<>
float collapse_accumulator<AVX512>(typename SimdType<AVX512>::vector vec) {
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
