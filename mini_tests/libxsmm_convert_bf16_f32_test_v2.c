#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <time.h>

#include <libxsmm.h>

float* generateRandomMatrix(int i_rows, int i_columns) {
  float* o_matrix = new float[i_rows * i_columns];
  int seed = 12;
  srand(seed);

  float l_minValue = -10.0;
  float l_maxValue = 10.0;

  for (int i = 0; i < i_rows; i++) {
    for (int j = 0; j < i_columns; j++) {
      float l_randomValue = l_minValue + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (l_maxValue - l_minValue)));
      o_matrix[i * i_columns + j] = l_randomValue;
    }
  }

  return o_matrix;
}

void subtractMatrices(const float* i_matrix_A, const float* i_matrix_B, float* o_matrix, int i_rows, int i_columns) {
  for (int i = 0; i < i_rows; i++) {
    for (int j = 0; j < i_columns; j++) {
      o_matrix[i * i_columns + j] = i_matrix_A[i * i_columns + j] - i_matrix_B[i * i_columns + j];
    }
  }
}

void print_hex_matrix(float *i_matrix, int rows, int columns) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < columns; ++c) {
        libxsmm_float_uint hybrid_in = {0};
        hybrid_in.f = i_matrix[r * columns + c];
        printf("0x%x ", hybrid_in.u);
    }
    printf("\n");
  }
}

void print_hex_and_compute_average(float *i_matrix_first_half, float *i_matrix_second_half, float *i_input, int i_rows, int i_columns) {
  // printf("Printing hex of the diff:\n");
  float l_diff_avg = 0;

  for (int l_r = 0; l_r < i_rows; ++l_r) {
    for (int l_c = 0; l_c < i_columns; ++l_c) {
      libxsmm_float_uint hybrid_in_1 = {0};
      libxsmm_float_uint hybrid_in_2 = {0};
      libxsmm_float_uint hybrid_in = {0};
      float l_diff = 0;

      hybrid_in_1.f = i_matrix_first_half[l_r * i_columns + l_c];
      hybrid_in_2.f = i_matrix_second_half[l_r * i_columns + l_c];
      hybrid_in.f = i_input[l_r * i_columns + l_c];
      l_diff = hybrid_in.f - (hybrid_in_1.f + hybrid_in_2.f);
      // printf("%f ", l_diff);
      l_diff_avg += abs(l_diff);
    }
    // printf("\n");
  }
  printf("Average difference: %f\n", l_diff_avg / (i_rows * i_columns));
}

int main() {
  int rows = 35;
  int columns = 20;
  int length = rows * columns;

  float* input = generateRandomMatrix(rows, columns);

  float* output = new float[rows * columns];
  float* second_half_input = new float[rows * columns];

  float* first_half_fp32 = new float[rows * columns];
  float* second_half_fp32 = new float[rows * columns];

  libxsmm_bfloat16 first_half_bf16[rows][columns];
  libxsmm_bfloat16 second_half_bf16[rows][columns];

  clock_t start_time = clock();

  libxsmm_truncate_convert_f32_bf16((const float*)input, (libxsmm_bfloat16*)first_half_bf16, length);
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)first_half_bf16, (float*)first_half_fp32, length);

  subtractMatrices((float*)input, (float*)first_half_fp32, (float*)second_half_input, rows , columns);

  libxsmm_rne_convert_fp32_bf16((const float*)second_half_input, (libxsmm_bfloat16*)second_half_bf16, length);
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)second_half_bf16, (float*)second_half_fp32, length);

  clock_t end_time = clock();
  double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  printf("Time spent: %lf seconds\n", time_spent);

  // print_hex_matrix(input, rows, columns);
  print_hex_and_compute_average(first_half_fp32, second_half_fp32, input, rows, columns);

  delete[] input;
  delete[] output;
  delete[] second_half_input;
  delete[] first_half_fp32;
  delete[] second_half_fp32;

  return 0;
}