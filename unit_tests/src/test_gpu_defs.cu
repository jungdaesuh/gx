#include "gtest/gtest.h"

#include <type_traits>

#include "device_funcs.h"

namespace {

void expect_complex_eq(const cuComplex& value, float real, float imag)
{
  EXPECT_FLOAT_EQ(value.x, real);
  EXPECT_FLOAT_EQ(value.y, imag);
}

void expect_complex_eq(const cuDoubleComplex& value, double real, double imag)
{
  EXPECT_DOUBLE_EQ(value.x, real);
  EXPECT_DOUBLE_EQ(value.y, imag);
}

__global__ void exercise_complex_operators(cuComplex* float_results,
                                           cuDoubleComplex* double_results)
{
  const cuComplex f = make_cuComplex(3.0f, 4.0f);
  const cuComplex g = make_cuComplex(1.0f, -2.0f);
  float_results[0] = f * g;
  float_results[1] = f / g;

  cuComplex float_product = f;
  float_product *= g;
  float_results[2] = float_product;

  cuComplex float_quotient = f;
  float_quotient /= g;
  float_results[3] = float_quotient;
  float_results[4] = 2.0 * f;
  float_results[5] = f * 2.0;
  float_results[6] = f / 2.0;
  float_results[7] = 2 * f;
  float_results[8] = f * 2;
  float_results[9] = f / 2;

  const cuDoubleComplex d = make_cuDoubleComplex(-2.0, 0.5);
  const cuDoubleComplex e = make_cuDoubleComplex(4.0, 3.0);
  double_results[0] = d * e;
  double_results[1] = d / e;

  cuDoubleComplex double_product = d;
  double_product *= e;
  double_results[2] = double_product;

  cuDoubleComplex double_quotient = d;
  double_quotient /= e;
  double_results[3] = double_quotient;
  double_results[4] = 2.0 * d;
  double_results[5] = d * 2.0;
  double_results[6] = d / 2.0;
  double_results[7] = 2 * d;
  double_results[8] = d * 2;
  double_results[9] = d / 2;
}

TEST(HipComplexOperators, UseComplexArithmeticOnHost)
{
  const cuComplex f = make_cuComplex(3.0f, 4.0f);
  const cuComplex g = make_cuComplex(1.0f, -2.0f);
  expect_complex_eq(f * g, 11.0f, -2.0f);
  expect_complex_eq(f / g, -1.0f, 2.0f);

  cuComplex float_product = f;
  float_product *= g;
  expect_complex_eq(float_product, 11.0f, -2.0f);

  cuComplex float_quotient = f;
  float_quotient /= g;
  expect_complex_eq(float_quotient, -1.0f, 2.0f);
  expect_complex_eq(2.0 * f, 6.0f, 8.0f);
  expect_complex_eq(f * 2.0, 6.0f, 8.0f);
  expect_complex_eq(f / 2.0, 1.5f, 2.0f);
  expect_complex_eq(2 * f, 6.0f, 8.0f);
  expect_complex_eq(f * 2, 6.0f, 8.0f);
  expect_complex_eq(f / 2, 1.5f, 2.0f);

  const cuDoubleComplex d = make_cuDoubleComplex(-2.0, 0.5);
  const cuDoubleComplex e = make_cuDoubleComplex(4.0, 3.0);
  expect_complex_eq(d * e, -9.5, -4.0);
  expect_complex_eq(d / e, -0.26, 0.32);

  cuDoubleComplex double_product = d;
  double_product *= e;
  expect_complex_eq(double_product, -9.5, -4.0);

  cuDoubleComplex double_quotient = d;
  double_quotient /= e;
  expect_complex_eq(double_quotient, -0.26, 0.32);
  expect_complex_eq(2.0 * d, -4.0, 1.0);
  expect_complex_eq(d * 2.0, -4.0, 1.0);
  expect_complex_eq(d / 2.0, -1.0, 0.25);
  expect_complex_eq(2 * d, -4.0, 1.0);
  expect_complex_eq(d * 2, -4.0, 1.0);
  expect_complex_eq(d / 2, -1.0, 0.25);
}

TEST(HipComplexOperators, UseComplexArithmeticOnDevice)
{
  cuComplex* float_results_device = nullptr;
  cuDoubleComplex* double_results_device = nullptr;
  ASSERT_EQ(cudaSuccess,
            cudaMalloc(reinterpret_cast<void**>(&float_results_device),
                       10 * sizeof(cuComplex)));
  ASSERT_EQ(cudaSuccess,
            cudaMalloc(reinterpret_cast<void**>(&double_results_device),
                       10 * sizeof(cuDoubleComplex)));

  exercise_complex_operators<<<1, 1>>>(float_results_device,
                                      double_results_device);
  ASSERT_EQ(cudaSuccess, cudaGetLastError());

  cuComplex float_results[10];
  cuDoubleComplex double_results[10];
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(float_results, float_results_device,
                       sizeof(float_results), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(double_results, double_results_device,
                       sizeof(double_results), cudaMemcpyDeviceToHost));

  for (int i : {0, 2}) {
    expect_complex_eq(float_results[i], 11.0f, -2.0f);
    expect_complex_eq(double_results[i], -9.5, -4.0);
  }
  for (int i : {1, 3}) {
    expect_complex_eq(float_results[i], -1.0f, 2.0f);
    expect_complex_eq(double_results[i], -0.26, 0.32);
  }
  for (int i : {4, 5, 7, 8}) {
    expect_complex_eq(float_results[i], 6.0f, 8.0f);
    expect_complex_eq(double_results[i], -4.0, 1.0);
  }
  expect_complex_eq(float_results[6], 1.5f, 2.0f);
  expect_complex_eq(double_results[6], -1.0, 0.25);
  expect_complex_eq(float_results[9], 1.5f, 2.0f);
  expect_complex_eq(double_results[9], -1.0, 0.25);

  EXPECT_EQ(cudaSuccess, cudaFree(float_results_device));
  EXPECT_EQ(cudaSuccess, cudaFree(double_results_device));
}

TEST(HipComplexOperators, AddScaledUsesRealDoubleCoefficients)
{
  const cuComplex first[] = {
      make_cuComplex(3.0f, 4.0f),
      make_cuComplex(-2.0f, 0.5f),
  };
  const cuComplex second[] = {
      make_cuComplex(-1.0f, 2.0f),
      make_cuComplex(4.0f, 3.0f),
  };

  cuComplex* first_device = nullptr;
  cuComplex* second_device = nullptr;
  cuComplex* result_device = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&first_device),
                                    sizeof(first)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&second_device),
                                    sizeof(second)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&result_device),
                                    sizeof(first)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(first_device, first, sizeof(first),
                                    cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(second_device, second, sizeof(second),
                                    cudaMemcpyHostToDevice));

  setdev_constants(2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1);
  add_scaled_kernel<<<1, 2>>>(result_device, 2.0, first_device, -0.5,
                              second_device, true);
  ASSERT_EQ(cudaSuccess, cudaGetLastError());

  cuComplex result[2];
  ASSERT_EQ(cudaSuccess, cudaMemcpy(result, result_device, sizeof(result),
                                    cudaMemcpyDeviceToHost));
  expect_complex_eq(result[0], 6.5f, 7.0f);
  expect_complex_eq(result[1], -6.0f, -0.5f);

  EXPECT_EQ(cudaSuccess, cudaFree(first_device));
  EXPECT_EQ(cudaSuccess, cudaFree(second_device));
  EXPECT_EQ(cudaSuccess, cudaFree(result_device));
}

#if HIPTENSOR_MAJOR_VERSION >= 2
TEST(HipTensorCompatibility, ExposesRocm7CutensorAliases)
{
  static_assert(CUTENSOR_VERSION == 20000,
                "ROCm 7 should select the hipTensor 2 compatibility API");
  static_assert(
      std::is_same<cutensorComputeDescriptor_t,
                   hiptensorComputeDescriptor_t>::value,
      "cutensor compute descriptors should map to hipTensor descriptors");
  static_assert(std::is_same<cutensorDataType_t, hiptensorDataType_t>::value,
                "cutensor data types should map to hipTensor data types");

  cutensorComputeDescriptor_t compute_type = CUTENSOR_COMPUTE_DESC_32F;
  cutensorDataType_t data_type = CUTENSOR_R_32F;
  EXPECT_EQ(HIPTENSOR_COMPUTE_DESC_32F, compute_type);
  EXPECT_EQ(HIPTENSOR_R_32F, data_type);

  auto create_tensor_descriptor = &cutensorCreateTensorDescriptor;
  auto reduce = &cutensorReduce;
  EXPECT_NE(nullptr, create_tensor_descriptor);
  EXPECT_NE(nullptr, reduce);
}
#endif

}  // namespace
