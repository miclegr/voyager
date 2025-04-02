#include "doctest.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sys/types.h>
#include <vector>

#include "TypedIndex.h"
#include "simd_utils.h"
#include "test_utils.h"


template <SIMD_ARCH Arch> 
void testSimdUtils() {
  using Simd = SimdType<Arch>;
  constexpr int N = Simd::floatsPerLine;
  float epsilon = 1e-5;

  SUBCASE("Test newAccumulator") {
    typename Simd::register_t __attribute__((aligned(64))) accReg = Simd::newAccumulator();
    const float *accPtr = reinterpret_cast<const float *>(&accReg);
    for (int i = 0; i < N; ++i) {
      CHECK(accPtr[i] == 0.0f);
    }
  }

  SUBCASE("Test loadAndConvertToFloat (float)") {
    std::vector<std::vector<float>> dataVec = randomVectors(1, N);
    const float *inputData = dataVec[0].data();

    typename Simd::register_t reg = Simd::loadAndConvertToFloat(inputData);
    const float *regPtr = reinterpret_cast<const float *>(&reg);

    for (int i = 0; i < N; ++i) {
      CHECK(regPtr[i] == doctest::Approx(inputData[i]).epsilon(epsilon));
    }
  }

  SUBCASE("Test loadAndConvertToFloat (int8_t)") {
    std::vector<std::vector<float>> dataVec = randomQuantizedVectors(1, N);
    const float *inputData = dataVec[0].data();

    typename Simd::register_t reg = Simd::loadAndConvertToFloat(inputData);
    const float *regPtr = reinterpret_cast<const float *>(&reg);

    for (int i = 0; i < N; ++i) {
      CHECK(regPtr[i] == doctest::Approx(static_cast<float>(inputData[i])).epsilon(epsilon));
    }
  }

   SUBCASE("Test loadAndConvertToFloat (E4M3)") {

       // We test all possible E4M3 in all possible positions of the SIMD vector
        std::vector<E4M3> inputData(256 + N);

        for (int i = 0; i < 256; ++i) {
            inputData[i] = E4M3(static_cast<uint8_t>(i));
        }

        for (int i = 200; i < 256 - N; ++i) {

          const E4M3 *inputDataI = inputData.data() + i;
          typename Simd::register_t reg = Simd::loadAndConvertToFloat(inputDataI);
          const float *regPtr = reinterpret_cast<const float *>(&reg);

          for (int j = 0; j < N; ++j) {
              float expected = static_cast<float>(inputData[i+j]);
              if (std::isnan(expected)) {
                  CHECK(doctest::IsNaN(regPtr[j]));
              } else {
                  CHECK( regPtr[j] == doctest::Approx(expected).epsilon(epsilon));
              }
          }

        }

       }

   SUBCASE("Test collapseAccumulator") {
    std::vector<std::vector<float>> dataVec = randomVectors(1, N);
    const float *inputData = dataVec[0].data();

    typename Simd::register_t __attribute__((aligned(64))) reg = Simd::loadAndConvertToFloat(inputData);

    float expectedSum = 0.0f;
    for (int i = 0; i < N; ++i) {
      expectedSum += inputData[i];
    }

    float collapsedSum = Simd::collapseAccumulator(reg);

    CHECK(collapsedSum == doctest::Approx(expectedSum).epsilon(epsilon));

  }
}

template <SIMD_ARCH arch> 
void testSimdDistanceCalulations() {

  using Simd = SimdType<arch>;
  constexpr int N = Simd::floatsPerLine;
  float epsilon = 1e-5;

  std::vector<SpaceType> spaceTypesSet = {
      SpaceType::Euclidean};
  std::vector<StorageDataType> storageTypesSet = {
      StorageDataType::Float8, StorageDataType::Float32, StorageDataType::E4M3};
  std::vector<int> numDimensionsSet = {256, 276};

  for (auto spaceType : spaceTypesSet) {
    for (auto storageType : storageTypesSet) {
      for (auto numDimensions: numDimensionsSet) {
        std::vector<std::vector<float>> v1, v2;

        if (storageType == StorageDataType::Float8 ||
            storageType == StorageDataType::E4M3) {
          v1 = randomQuantizedVectors(1, numDimensions);
          v2 = randomQuantizedVectors(1, numDimensions);
        } else if (storageType == StorageDataType::Float32) {
          v1 = randomVectors(1, numDimensions);
          v2 = randomVectors(1, numDimensions);
        }

        SUBCASE("Test L2Sqr") {
          CAPTURE(spaceType);
          CAPTURE(storageType);

          float distance = hnswlib::L2Sqr<float>(v1[0].data(), v2[0].data(), N);
          float distanceSimd;
          if (numDimensions % N == 0) {
          distanceSimd = hnswlib::L2SqrSimd<arch>(v1[0].data(), v2[0].data(), N);
          } else {
          distanceSimd = hnswlib::L2SqrAtLeastSimd<arch>(v1[0].data(), v2[0].data(), N);
          }
          CHECK(distance == doctest::Approx(distanceSimd).epsilon(epsilon));
        }
      }
    }
  }

}

TEST_SUITE_BEGIN("SIMD");

TEST_CASE("Test SIMD Utilities") {
  SIMD_ARCH arch = getX86SimdArch();

  switch (arch) {
  case SIMD_ARCH::AVX512:
    testSimdUtils<SIMD_ARCH::AVX512>();
    break;
  case SIMD_ARCH::AVX2:
    testSimdUtils<SIMD_ARCH::AVX512>();
    break;
  case SIMD_ARCH::SSE:
    testSimdUtils<SSE>();
    break;
  default:
    FAIL("Unknown or no SIMD architecture detected!");
    break;
  }
}

TEST_CASE("Test SIMD distance calculations") {
  SIMD_ARCH arch = getX86SimdArch();

  switch (arch) {
  case SIMD_ARCH::AVX512:
    testSimdDistanceCalulations<AVX512>();
    break;
  case SIMD_ARCH::AVX2:
    testSimdDistanceCalulations<AVX2>();
    break;
  case SIMD_ARCH::SSE:
    testSimdDistanceCalulations<SSE>();
    break;
  default:
    FAIL("Unknown or no SIMD architecture detected!");
    break;
  }
}

TEST_SUITE_END();
