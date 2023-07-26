#include "ConvertMatrix.h"

void rat_gemm::backend::ConvertMatrix::generateRandomMatrix(float * io_matrix, int i_s) {
  int seed = 12;
  srand(seed);

  float l_minValue = -10.0;
  float l_maxValue = 10.0;
  float l_randomValue; 
  for (int i = 0; i < i_s; i++) {
      l_randomValue = l_minValue + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (l_maxValue - l_minValue)));
      io_matrix[i] = l_randomValue;
  }
}

void rat_gemm::backend::ConvertMatrix::diff(float *i_matrix, const float *i_matrix_h1, const float *i_matrix_h2, float* o_diff, int i_s) {
  for (int i = 0; i < i_s; ++i) {
    libxsmm_float_uint hybrid_in_1 = {0};
    libxsmm_float_uint hybrid_in_2 = {0};
    libxsmm_float_uint hybrid_in = {0};

    hybrid_in_1.f = i_matrix_h1[i];
    hybrid_in_2.f = i_matrix_h2[i];
    hybrid_in.f = i_matrix[i];
    o_diff[i] = hybrid_in.f - (hybrid_in_1.f + hybrid_in_2.f);
  }
}

void rat_gemm::backend::ConvertMatrix::convert_fp32_two_bf16(const float* i_matrix,
                                                             libxsmm_bfloat16* o_matrix_bf16_h1,
                                                             libxsmm_bfloat16* o_matrix_bf16_h2,
                                                             int i_s){
  float* l_first_half_fp32 = new float[i_s];
  float* l_second_half_input = new float[i_s];

  libxsmm_truncate_convert_f32_bf16((const float*)i_matrix, (libxsmm_bfloat16*)o_matrix_bf16_h1, i_s);
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)o_matrix_bf16_h1, (float*)l_first_half_fp32, i_s);

  for (int i = 0; i < i_s; i++) {
      l_second_half_input[i] = i_matrix[i] - l_first_half_fp32[i];
  }

  libxsmm_rne_convert_fp32_bf16((const float*)l_second_half_input, (libxsmm_bfloat16*)o_matrix_bf16_h2, i_s);

  delete[] l_first_half_fp32;
  delete[] l_second_half_input;
}