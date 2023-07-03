#include "headers.hpp"
#include <libxsmm.h>
#include <chrono>

void reference_unpack(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *in, libxsmm_bfloat16 *out_lo, libxsmm_bfloat16 *out_hi) {
  libxsmm_blasint i, j;
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      libxsmm_bfloat16_f32 bf16_hp;
      bf16_hp.f = in[j * ld + i];
      out_lo[j * ld + i] = bf16_hp.i[2];
      out_hi[j * ld + i] = bf16_hp.i[1];
    }
  }
}

void reference_pack(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *out, libxsmm_bfloat16 *in_lo, libxsmm_bfloat16 *in_hi) {
  libxsmm_blasint i, j;
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      libxsmm_bfloat16_f32 bf16_hp;
      bf16_hp.i[2] = in_lo[j * ld + i];
      bf16_hp.i[1] = in_hi[j * ld + i];
      out[j * ld + i] = bf16_hp.f;
    }
  }
}

int main() {
  int rows = 1;
  int columns = 10;
  
  float** input = generateRandomMatrix(rows, columns);
  
  float output[rows][columns];

  libxsmm_bfloat16 first_half_bf16[rows][columns];
  libxsmm_bfloat16 second_half_bf16[rows][columns];

  // Measure the start time
  auto start = std::chrono::high_resolution_clock::now();

  reference_unpack(columns, rows, 0, (float*)input, (libxsmm_bfloat16*)first_half_bf16, (libxsmm_bfloat16*)second_half_bf16);

  // Measure the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the elapsed time
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Convert back to fp32 to check the difference
  reference_pack(columns, rows, 0, (float*)output, (libxsmm_bfloat16*)first_half_bf16, (libxsmm_bfloat16*)second_half_bf16);

  // Print values
  print((float*)input, (float*)output, rows , columns);

  // Print the elapsed time in microseconds
  std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

  // Free the allocated memory
  freeMatrix(input, rows);

  return 0;
}
