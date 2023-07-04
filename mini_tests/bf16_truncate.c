#include <libxsmm.h>
#include "headers.hpp"
#include <chrono>

int main() {
  int rows = 10;
  int columns = 2000;
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
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)first_half_bf16, (float*)first_half_fp32, length);

  // Print the original bfloat16 values and truncated bfloat16 values  
  subtractMatrices((float*)input, (float*)first_half_fp32, (float*)second_half_input, rows , columns);
  
  libxsmm_rne_convert_fp32_bf16((const float*)second_half_input, (libxsmm_bfloat16*)second_half_bf16, length);
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)second_half_bf16, (float*)second_half_fp32, length);

  // Measure the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the elapsed time
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  addMatrices((const float*) first_half_fp32, (const float*) second_half_fp32, (float*)output, rows, columns);
  print(input, output,  rows , columns);

  // Print the elapsed time in microseconds
  std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;


  delete[] input;
  delete[] output;
  delete[] second_half_input;
  delete[] first_half_fp32;
  delete[] second_half_fp32;

  return 0;
}
