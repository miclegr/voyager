#include "doctest.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sys/types.h>
#include <vector>
#include <tuple>
#include <variant>

#include "TypedIndex.h"
#include "simd_utils.h"
#include "test_utils.h"

template <SIMD_ARCH Arch> 
void testSimdUtils() {
  using Simd = SimdType<Arch>;
  constexpr int N = Simd::floatsPerLine;
  float epsilon = 1e-5;

  SUBCASE("Test newAccumulator") {
    std::vector<float> accReg = Simd::registerToVectorWrapper(Simd::newAccumulator);
    const float *accPtr = accReg.data();
    for (int i = 0; i < N; ++i) {
      CHECK(accPtr[i] == 0.0f);
    }
  }

  SUBCASE("Test loadAndConvertToFloat (float)") {
    std::vector<std::vector<float>> dataVec = randomVectors(1, N);
    const float *inputData = dataVec[0].data();

    auto loadFloatAndConvertToFloat = Simd:: template loadAndConvertToFloat<float>;
    std::vector<float> reg = Simd::registerToVectorWrapper(loadFloatAndConvertToFloat,inputData);
    const float *regPtr = reg.data();

    for (int i = 0; i < N; ++i) {
      CHECK(regPtr[i] == doctest::Approx(inputData[i]).epsilon(epsilon));
    }
  }

  SUBCASE("Test loadAndConvertToFloat (int8_t)") {
    std::vector<std::vector<int8_t>> dataVec= randomQuantizedVectors<int8_t>(1, N);
    const int8_t *inputData = dataVec[0].data();

    auto loadInt8AndConvertToFloat = Simd:: template loadAndConvertToFloat<int8_t>;
    std::vector<float> reg = Simd::registerToVectorWrapper(loadInt8AndConvertToFloat, inputData);
    const float *regPtr = reg.data();

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

        auto loadE4M3AndConvertToFloat = Simd:: template loadAndConvertToFloat<E4M3>;

        for (int i = 200; i < 256 - N; ++i) {

          const E4M3 *inputDataI = inputData.data() + i;
          std::vector<float> reg = Simd::registerToVectorWrapper(loadE4M3AndConvertToFloat, inputDataI);
          const float *regPtr = reg.data();

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

    float expectedSum = 0.0f;
    for (int i = 0; i < N; ++i) {
      expectedSum += inputData[i];
    }

    auto accumulator = *(reinterpret_cast<const typename Simd::register_t*>(inputData));
    float collapsedSum = Simd::collapseAccumulator(accumulator);

    CHECK(collapsedSum == doctest::Approx(expectedSum).epsilon(epsilon));

  }
}

template <SIMD_ARCH arch> 
void testSimdDistanceCalulations() {

  using Simd = SimdType<arch>;
  using FloatVectorVector = std::vector<std::vector<float>>;
  using Int8VectorVector = std::vector<std::vector<int8_t>>;
  using E4M3VectorVector = std::vector<std::vector<E4M3>>;
  using FloatDistanceFn = std::function<float(const float*, const float*, size_t)>;
  using Int8DistanceFn = std::function<float(const int8_t*, const int8_t*, size_t)>;
  using E4M3DistanceFn= std::function<float(const E4M3*, const E4M3*, size_t)>;


  constexpr int N = Simd::floatsPerLine;
  float epsilon = 1e-5;

  std::vector<SpaceType> spaceTypesSet = {
      SpaceType::Euclidean, SpaceType::InnerProduct};
  std::vector<StorageDataType> storageTypesSet = {
      StorageDataType::Float8, StorageDataType::Float32, StorageDataType::E4M3
  };
  std::vector<int> numDimensionsSet = {256, 276};


  std::variant<
    std::tuple<FloatVectorVector, FloatVectorVector, FloatDistanceFn, FloatDistanceFn, FloatDistanceFn>,
    std::tuple<Int8VectorVector, Int8VectorVector, Int8DistanceFn, Int8DistanceFn, Int8DistanceFn>,
    std::tuple<E4M3VectorVector, E4M3VectorVector, E4M3DistanceFn, E4M3DistanceFn, E4M3DistanceFn>
    > testData;

  for (auto spaceType : spaceTypesSet) {
    for (auto storageType : storageTypesSet) {
      for (auto numDimensions: numDimensionsSet) {

        if (storageType == StorageDataType::Float8) {
          Int8VectorVector v1 = randomQuantizedVectors<int8_t>(1, numDimensions);
          Int8VectorVector v2 = randomQuantizedVectors<int8_t>(1, numDimensions);
          Int8DistanceFn NonSimdDistanceFn, SimdDistanceFn, SimdDistanceAtLeastFn;
          if (spaceType == SpaceType::Euclidean) {
            NonSimdDistanceFn = hnswlib::L2Sqr<float, int8_t, 1, std::ratio<1,1>>;
            SimdDistanceFn = hnswlib::L2SqrSimd<arch, int8_t, std::ratio<1,1>>;
            SimdDistanceAtLeastFn = hnswlib::L2SqrAtLeastSimd<arch, int8_t, std::ratio<1,1>>;
          } else {
            NonSimdDistanceFn = hnswlib::InnerProduct<float, int8_t, 1, std::ratio<1,1>>;
            SimdDistanceFn = hnswlib::InnerProductSimd<arch, int8_t, std::ratio<1,1>>;
            SimdDistanceAtLeastFn = hnswlib::InnerProductAtLeastSimd<arch, int8_t, std::ratio<1,1>>;
          }
          testData = std::make_tuple(v1, v2, NonSimdDistanceFn, SimdDistanceFn, SimdDistanceAtLeastFn);
        } else if (storageType == StorageDataType::E4M3) {
          E4M3VectorVector v1 = randomQuantizedVectors<E4M3>(1, numDimensions);
          E4M3VectorVector v2 = randomQuantizedVectors<E4M3>(1, numDimensions);
          E4M3DistanceFn NonSimdDistanceFn, SimdDistanceFn, SimdDistanceAtLeastFn;
          if (spaceType == SpaceType::Euclidean) {
            NonSimdDistanceFn = hnswlib::L2Sqr<float, E4M3, 1, std::ratio<1,1>>;
            SimdDistanceFn = hnswlib::L2SqrSimd<arch, E4M3, std::ratio<1,1>>;
            SimdDistanceAtLeastFn = hnswlib::L2SqrAtLeastSimd<arch, E4M3, std::ratio<1,1>>;
          } else {
            NonSimdDistanceFn = hnswlib::InnerProduct<float, E4M3, 1, std::ratio<1,1>>;
            SimdDistanceFn = hnswlib::InnerProductSimd<arch, E4M3, std::ratio<1,1>>;
            SimdDistanceAtLeastFn = hnswlib::InnerProductAtLeastSimd<arch, E4M3, std::ratio<1,1>>;
          }
          testData = std::make_tuple(v1, v2, NonSimdDistanceFn, SimdDistanceFn, SimdDistanceAtLeastFn);
        } else if (storageType == StorageDataType::Float32) {
          FloatVectorVector v1 = randomVectors(1, numDimensions);
          FloatVectorVector v2 = randomVectors(1, numDimensions);
          FloatDistanceFn NonSimdDistanceFn, SimdDistanceFn, SimdDistanceAtLeastFn;
          if (spaceType == SpaceType::Euclidean) {
            NonSimdDistanceFn = hnswlib::L2Sqr<float, float, 1, std::ratio<1,1>>;
            SimdDistanceFn = hnswlib::L2SqrSimd<arch, float, std::ratio<1,1>>;
            SimdDistanceAtLeastFn = hnswlib::L2SqrAtLeastSimd<arch, float, std::ratio<1,1>>;
          } else {
            NonSimdDistanceFn = hnswlib::InnerProduct<float, float, 1, std::ratio<1,1>>;
            SimdDistanceFn = hnswlib::InnerProductSimd<arch, float, std::ratio<1,1>>;
            SimdDistanceAtLeastFn = hnswlib::InnerProductAtLeastSimd<arch, float, std::ratio<1,1>>;
          }
          testData = std::make_tuple(v1, v2, NonSimdDistanceFn, SimdDistanceFn, SimdDistanceAtLeastFn);
        }
        
        SUBCASE("Test Distance") {
          CAPTURE(spaceType);
          CAPTURE(storageType);
          CAPTURE(numDimensions);

          float distance = std::visit(
            [&](auto&& testData) {
              auto&& v1 = std::get<0>(testData);
              auto&& v2 = std::get<1>(testData);
              auto&& fn = std::get<2>(testData);
              return fn(v1[0].data(), v2[0].data(), numDimensions);
            },
            testData);

          float distanceSimd = std::visit(
            [&](auto&& testData) {
              auto&& v1 = std::get<0>(testData);
              auto&& v2 = std::get<1>(testData);
              auto&& fn = std::get<3>(testData);
              auto&& fnAtLeast = std::get<4>(testData);
              if (numDimensions % N == 0)
                return fn(v1[0].data(), v2[0].data(), numDimensions);
              else
                return fnAtLeast(v1[0].data(), v2[0].data(), numDimensions);
            },
            testData);
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
    testSimdUtils<SIMD_ARCH::AVX2>();
    break;
  case SIMD_ARCH::SSE:
    testSimdUtils<SIMD_ARCH::SSE>();
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
    testSimdDistanceCalulations<SIMD_ARCH::AVX512>();
    break;
  case SIMD_ARCH::AVX2:
    testSimdDistanceCalulations<SIMD_ARCH::AVX2>();
    break;
  case SIMD_ARCH::SSE:
    testSimdDistanceCalulations<SIMD_ARCH::SSE>();
    break;
  default:
    FAIL("Unknown or no SIMD architecture detected!");
    break;
  }
}

TEST_SUITE_END();
