#include <libxsmm.h>
#include "headers.hpp"
#include <chrono>
/////////////////////////Wrong//////////////////////////////////////


// #include <cstdint>
// #include <cstddef>

void bf16ToFP32(const libxsmm_bfloat16* bf16Array, float* fp32Array, size_t rows, size_t columns) {
  const unsigned int length = rows * columns;

  for (unsigned int i = 0; i < length; ++i) {
    libxsmm_bfloat16 bf16Value = bf16Array[i];

    // Extracting the sign, exponent, and fraction bits from the BF16 value
    uint32_t sign = (bf16Value & 0x8000) << 17;
    std::cout<< sign <<std::endl;
    uint32_t exponent = (bf16Value & 0x7F800000) << 15;
    std::cout<< exponent <<std::endl;
    uint32_t fraction = (bf16Value & 0x7F000000);
    std::cout<< fraction <<std::endl;

    // Building the FP32 value with zero upscaling
    uint32_t fp32Value = sign | exponent | fraction;
    std::cout<< fp32Value <<std::endl;

    // Casting the resulting bits to the FP32 format and storing it in the FP32 array
    fp32Array[i] = *reinterpret_cast<float*>(&fp32Value);
    std::cout<< fp32Array[i] <<std::endl;
  }
}


int main() {
  int rows = 1;
  int columns = 10;
  int length = rows * columns;
  
  float* input = generateRandomMatrix(rows, columns);
  float* output = generateRandomMatrix(rows, columns);
  float* second_half_input = generateRandomMatrix(rows, columns);

  float* first_half_fp32 = generateRandomMatrix(rows, columns);
  float* second_half_fp32 = generateRandomMatrix(rows, columns);

  libxsmm_bfloat16 first_half_bf16[rows][columns];
  libxsmm_bfloat16 second_half_bf16[rows][columns];

  // Measure the start time
  auto start = std::chrono::high_resolution_clock::now();
  
  libxsmm_truncate_convert_f32_bf16((const float*)input, (libxsmm_bfloat16*)first_half_bf16, length);

  bf16ToFP32((libxsmm_bfloat16*)first_half_bf16, first_half_fp32, rows, columns);
  print(input, first_half_fp32,  rows , columns);

  subtractMatrices((float*)input, (float*)first_half_fp32, (float*)second_half_input, rows , columns);
  
  libxsmm_rne_convert_fp32_bf16((const float*)second_half_input, (libxsmm_bfloat16*)second_half_bf16, length);
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)second_half_bf16, (float*)second_half_fp32, length);

  // Measure the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the elapsed time
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  addMatrices((const float*) first_half_fp32, (const float*) second_half_fp32, (float*)output, rows, columns);
  // print(input, output,  rows , columns);

  // Print the elapsed time in microseconds
  std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;


  delete[] input;
  delete[] output;
  delete[] second_half_input;
  delete[] first_half_fp32;
  delete[] second_half_fp32;

  return 0;
}
