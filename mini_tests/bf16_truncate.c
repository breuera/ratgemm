#include <libxsmm.h>
#include "headers.hpp"
#include <chrono>

void upconvert_bf16(libxsmm_blasint i_rows, libxsmm_blasint i_columns, libxsmm_bfloat16* in, float* out) {
  libxsmm_blasint i, j;
  for (i = 0; i < i_rows; i++) {
    for (j = 0; j < i_columns; j++) {
      libxsmm_bfloat16_f32 bf16_hp /* = { 0 }*/;
      bf16_hp.i[1] = in[i * i_columns + j];
      bf16_hp.i[0] = 0;
      out[i * i_columns + j] = bf16_hp.f;
    }
  }
}

int main() {
  int rows = 10;
  int columns = 1000;
  int length = rows * columns;
  
  float* input = generateRandomMatrix(rows, columns);
  // for (int i = 0; i < rows; ++i) {
  //   for (int j = 0; j < columns; ++j) {
  //     std::cout << "Input: " << input[i * columns + j] << std::endl;
  //   }
  // }

  std::cout << "input: " << input[0] << std::endl; // 1 10000001 00100000 01010011 0110011
                                       // Output   // 1 10000001 00100000 01010011 1000111


  float* output = new float[rows * columns];
  float* second_half_input = new float[rows * columns];

  float* first_half_fp32 = new float[rows * columns];
  float* second_half_fp32 = new float[rows * columns];

  libxsmm_bfloat16 first_half_bf16[rows][columns];
  libxsmm_bfloat16 second_half_bf16[rows][columns];

  // Measure the start time
  auto start = std::chrono::high_resolution_clock::now();
  
  libxsmm_truncate_convert_f32_bf16((const float*)input, (libxsmm_bfloat16*)first_half_bf16, length);
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)first_half_bf16, (float*)first_half_fp32, length);
  // upconvert_bf16(rows , columns,(libxsmm_bfloat16*)first_half_bf16, (float*)first_half_fp32); // x3 slower
  // print(input, first_half_fp32, rows , columns);
  std::cout << "first_half_bf16: " << *first_half_bf16[0] << std::endl; // 1 10000001 0010000
  std::cout << "first_half_fp32: " << first_half_fp32[0] << std::endl;  // 1 10000001 0010000 0000000000000000

  

  // Print the original bfloat16 values and truncated bfloat16 values  
  subtractMatrices((float*)input, (float*)first_half_fp32, (float*)second_half_input, rows , columns);
  std::cout << "second_half_input: " << second_half_input[0] << std::endl; // 1 01110111 0100110 1011110000000001
  
  libxsmm_rne_convert_fp32_bf16((const float*)second_half_input, (libxsmm_bfloat16*)second_half_bf16, length);
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)second_half_bf16, (float*)second_half_fp32, length);

  std::cout << "second_half_bf16: " << *second_half_bf16[0] << std::endl; // 1 01110111 0100111 + rounding nearest even
  std::cout << "second_half_fp32: " << second_half_fp32[0] << std::endl;  // 1 01110111 0100111 0000000000001010

  // upconvert_bf16(rows , columns,(libxsmm_bfloat16*)second_half_bf16, (float*)second_half_fp32); // x3 slower

  // Measure the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the elapsed time
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  addMatrices((const float*) first_half_fp32, (const float*) second_half_fp32, (float*)output, rows, columns);
  // print(input, output, rows , columns);
  std::cout << "output: " << output[0] << std::endl; // 1 10000001 00100000010100111000111


  // Print the elapsed time in microseconds
  std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

  delete[] input;
  delete[] output;
  delete[] second_half_input;
  delete[] first_half_fp32;
  delete[] second_half_fp32;

  return 0;
}
