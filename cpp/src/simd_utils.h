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
#ifdef _WIN32
#include <intrin.h>
#endif

template<typename T, typename... Args>
using FuncPtr = T(*)(Args...);

template<typename T, typename... Args>
FuncPtr<T, Args...> x86_simd_dispatch(FuncPtr<T, Args...> sse_version, FuncPtr<T, Args...> avx_version, FuncPtr<T, Args...> avx512_version) {
#ifdef _WIN32
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] >= 7) {
        __cpuid(cpuInfo, 7);
        if (cpuInfo[1] & (1 << 16)) {
            return avx512_version;
        }
    }
    __cpuid(cpuInfo, 1);
    if (cpuInfo[2] & (1 << 28)) {
        return avx_version;
    }
    return sse_version;
#else
    if (__builtin_cpu_supports("avx512f")) {
        return avx512_version;
    } else if (__builtin_cpu_supports("avx")) {
        return avx_version;
    } else {
        return sse_version;
    }
#endif
    return sse_version;
}


// Example SSE, AVX, and AVX512 functions
template<typename T>
T sse_version(T a, T b) {
    std::cout << "Using SSE version" << std::endl;
    return a + b;
}

template<typename T>
T avx_version(T a, T b) {
    std::cout << "Using AVX version" << std::endl;
    return a + b;
}

template<typename T>
T avx512_version(T a, T b) {
    std::cout << "Using AVX512 version" << std::endl;
    return a + b;
}

int main() {
    int a = 5, b = 10;
    auto func = x86_simd_dispatch<int>(sse_version<int>, avx_version<int>, avx512_version<int>);
    int result = func(a, b);
    std::cout << "Result: " << result << std::endl;
    return 0;
}

