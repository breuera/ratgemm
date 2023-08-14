#include "headers.hpp"
#include <libxsmm.h>
#include <chrono>

void reference_unpack(libxsmm_blasint i_rows, libxsmm_blasint i_columns, float *in, libxsmm_bfloat16 *out_lo, libxsmm_bfloat16 *out_hi) {
  libxsmm_blasint i, j;
  for (i = 0; i < i_rows; i++) {
    for (j = 0; j < i_columns; j++) {
      libxsmm_bfloat16_f32 bf16_hp;
      bf16_hp.f = in[i * i_columns + j];
      out_lo[i * i_columns + j] = bf16_hp.i[2];
      out_hi[i * i_columns + j] = bf16_hp.i[1];
    }
  }
}

void reference_pack(libxsmm_blasint i_rows, libxsmm_blasint i_columns, float *out, libxsmm_bfloat16 *in_lo, libxsmm_bfloat16 *in_hi) {
  libxsmm_blasint i, j;
  for (i = 0; i < i_rows; i++) {
    for (j = 0; j < i_columns; j++) {
      libxsmm_bfloat16_f32 bf16_hp;
      bf16_hp.i[2] = in_lo[i * i_columns + j];
      bf16_hp.i[1] = in_hi[i * i_columns + j];
      out[i * i_columns + j] = bf16_hp.f;
    }
  }
}

int main() {
  int rows = 10;
  int columns = 1000;
  
  float* input = generateRandomMatrix(rows, columns);
  float* output = new float[rows * columns];

  libxsmm_bfloat16 first_half_bf16[rows][columns];
  libxsmm_bfloat16 second_half_bf16[rows][columns];

  // Measure the start time
  auto start = std::chrono::high_resolution_clock::now();
  // Convert one fp32 to two bf16
  reference_unpack(rows, columns, (float*)input, (libxsmm_bfloat16*)first_half_bf16, (libxsmm_bfloat16*)second_half_bf16);

  // Measure the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the elapsed time
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Convert back to fp32 from two bf16 to check the difference
  reference_pack(rows, columns, (float*)output, (libxsmm_bfloat16*)first_half_bf16, (libxsmm_bfloat16*)second_half_bf16);

  // Print values
  print(input, output, rows, columns);

  // Print the elapsed time in microseconds
  std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

  delete[] input;
  delete[] output;

  return 0;
}
