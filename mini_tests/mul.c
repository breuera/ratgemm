#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <iomanip> // For setting precision when printing floating-point values
#include <random> // For random number generation
#include <libxsmm.h>

std::vector<int64_t> l_nz_idx;

float bfloat16_to_float(libxsmm_bfloat16 bf16_value) {
    uint16_t bf16_bits = static_cast<uint16_t>(bf16_value);
    uint32_t fp32_bits = ((bf16_bits & 0x7FFF) << 16) | ((bf16_bits & 0x8000) << 16) | ((bf16_bits & 0x7F) << 13);
    float float_value;
    memcpy(&float_value, &fp32_bits, sizeof(float));
    return float_value;
}


void RowNonZero(const std::vector<libxsmm_bfloat16>& i_vec,
                int i_m,
                int i_n,
                std::vector<int64_t>& o_nz_idx) {

  o_nz_idx.clear();

  for (int i = 0; i < i_m; ++i) {
    bool l_rowHasNonZero = false;
    for (int j = 0; j < i_n; ++j) {
      if (i_vec[i * i_n + j] != 0) {
          l_rowHasNonZero = true;
          break;
      }
    }
    if (l_rowHasNonZero) {
      o_nz_idx.push_back(i);
    }
  }
}

void pad_rows(const std::vector<libxsmm_bfloat16>& i_vec_1,
              const std::vector<libxsmm_bfloat16>& i_vec_2,
              const std::vector<int64_t>& i_indices,
              std::vector<libxsmm_bfloat16>& o_mat_padded,
              const int64_t i_m,
              const int64_t i_n) {

  // Copy main values
  for (int64_t l_n = 0; l_n < i_n; l_n++) {
    for (int64_t l_m = 0; l_m < i_m; l_m++) {
      o_mat_padded.push_back(i_vec_1[l_m + i_m * l_n]);
    }
  }
  // Append non-zero row
  for (std::size_t l_co_in = 0; l_co_in < i_indices.size(); l_co_in++) {
    for (int64_t l_n = 0; l_n < i_n; l_n++) {
      int64_t l_id_in = i_n * i_indices[l_co_in] + l_n;
      o_mat_padded.push_back(i_vec_2[l_id_in]);
    }
  }
}

void pad_cols(const std::vector<libxsmm_bfloat16>& io_vec_1,
              const std::vector<libxsmm_bfloat16>& i_vec_2,
              const std::vector<int64_t>& i_indices,
              std::vector<libxsmm_bfloat16>& o_mat_padded,
              const int64_t i_m,
              const int64_t i_n) {

  // Copy main values
  for (int64_t l_m = 0; l_m < i_m; l_m++) {
    for (int64_t l_n = 0; l_n < i_n; l_n++) {
      o_mat_padded.push_back(io_vec_1[l_m * i_n + l_n]);
    }
    // Append non-zero row
    for (std::size_t l_co_in = 0; l_co_in < i_indices.size(); l_co_in++) {
      int64_t l_id_in = i_indices[l_co_in] + (l_m * i_n);
      o_mat_padded.push_back(i_vec_2[l_id_in]);
    }
  }
}

void bf16_gemm_one(libxsmm_bfloat16* i_dof,
                   libxsmm_bfloat16* i_stiff,
                   float* o_result){ 

   for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    for( int64_t l_m = 0; l_m < 2; l_m++ ) {
      for( int64_t l_n = 0; l_n < 2; l_n++ ) {
        float acc = 0.0f;
        for( int64_t l_k = 0; l_k < (4 + l_nz_idx.size()); l_k++ ) {
          acc += bfloat16_to_float(i_dof[l_n * (4 + l_nz_idx.size()) + l_k]) * bfloat16_to_float(i_stiff[l_di * 2 + l_k * 2 * 3 + l_m]);

          std::cout << l_n * (4 + l_nz_idx.size()) + l_k << " * " << l_di * 2 + l_k * 2 * 3 + l_m << " = " <<  bfloat16_to_float(i_dof[l_n * (4 + l_nz_idx.size()) + l_k]) << " * " << bfloat16_to_float(i_stiff[l_di * 2 + l_k * 2 * 3 + l_m]) << std::endl; 
        }
        o_result[l_di * 2 * 2 + l_n * 2 + l_m] = acc;
        std::cout << " = " << l_di * 2 * 2 + l_n * 2 + l_m; 
        std::cout << " = " << acc << std::endl; 
      }
    }
  }
}

void convert_fp32_two_bf16(const float* i_matrix,
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

  libxsmm_truncate_convert_f32_bf16((const float*)l_second_half_input, (libxsmm_bfloat16*)o_matrix_bf16_h2, i_s);

  delete[] l_first_half_fp32;
  delete[] l_second_half_input;
}






int main(){
  float l_dofs[2][4] = { 0 };

  for( int64_t l_qt = 0; l_qt < 2; l_qt++ ) {
    for( int64_t l_md = 0; l_md < 4; l_md++ ) {
      l_dofs[l_qt][l_md] = (float) (rand()) / (float) (RAND_MAX);
    }
  }

  double new_stiff_test[3][4][2] = {
      {
        {0, -2.3},
        {0, 0},
        {0, 0},
        {0.2, 0}
      },
      {
        {0, 0},
        {0, 0},
        {0, 0},
        {0.3, 0}
      },
      {
        {0, -1.0},
        {0, 0},
        {0, 0},
        {-1.0, 0}
      }
    };

  float l_stiff_fp32[3][4][2];

  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    for( int64_t l_n = 0; l_n < 4; l_n++ ) {
      for( int64_t l_m = 0; l_m < 2; l_m++ ) {
        l_stiff_fp32[l_di][l_n][l_m] = new_stiff_test[l_di][l_n][l_m];
      }
    }
  }

  float l_stiff_single[4][3*2];

  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    for( int64_t l_n = 0; l_n < 4; l_n++ ) {
      for( int64_t l_m = 0; l_m < 2; l_m++ ) {
        int64_t l_m_single = l_di*2 + l_m;
        l_stiff_single[l_n][l_m_single] = l_stiff_fp32[l_di][l_n][l_m];
      }
    }
  }

  // // Display the converted to 2d
  // for (int i = 0; i < 4; i++) {
  //   for (int j = 0; j < 3*2; j++) {
  //     std::cout << l_stiff_single[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  libxsmm_bfloat16* l_dof_1 = new libxsmm_bfloat16[2 * 4];
  libxsmm_bfloat16* l_dof_2 = new libxsmm_bfloat16[2 * 4];


  libxsmm_bfloat16* l_stiff_1 = new libxsmm_bfloat16[4 * 3*2];
  libxsmm_bfloat16* l_stiff_2 = new libxsmm_bfloat16[4 * 3*2];

  convert_fp32_two_bf16((const float *)l_stiff_single, l_stiff_1, l_stiff_2, 4 * 3*2);
  convert_fp32_two_bf16((const float *)l_dofs, l_dof_1, l_dof_2, 2 * 4);

  std::vector<libxsmm_bfloat16> l_stiff_1_vec(l_stiff_1, l_stiff_2 + 4 * 3*2);
  std::vector<libxsmm_bfloat16> l_stiff_2_vec(l_stiff_2, l_stiff_2 + 4 * 3*2);

  // Display the values in float converted to bf16

  // l_nz_idx = {2};
  RowNonZero(l_stiff_2_vec, 4, 3*2, l_nz_idx);

  std::vector<libxsmm_bfloat16> l_stiff_padded;
  pad_rows(l_stiff_1_vec, l_stiff_2_vec, l_nz_idx, l_stiff_padded, 4, 3*2);


  std::vector<libxsmm_bfloat16> l_dof_1_vec(l_dof_1, l_dof_1 + 2 * 4);
  std::vector<libxsmm_bfloat16> l_dof_2_vec(l_dof_2, l_dof_2 + 2 * 4);

  std::vector<libxsmm_bfloat16> l_dof_1_padded;
  std::vector<libxsmm_bfloat16> l_dof_2_padded;

  pad_cols(l_dof_1_vec, l_dof_1_vec, l_nz_idx, l_dof_1_padded, 2, 4);
  pad_cols(l_dof_2_vec, l_dof_2_vec, l_nz_idx, l_dof_2_padded, 2, 4);

  // for (int i = 0; i < 2; ++i) {
  //   for (int j = 0; j < (4 + l_nz_idx.size()); ++j) {
  //     std::cout << bfloat16_to_float(l_dof_1_padded[i * (4 + l_nz_idx.size()) + j]) << "   ";
  //   }
  //   std::cout << std::endl;
  // }

  float l_result[2][3][2] = { 0 };

  auto start = std::chrono::high_resolution_clock::now();

  bf16_gemm_one((libxsmm_bfloat16 *)l_dof_1_padded.data(),
                (libxsmm_bfloat16 *)l_stiff_padded.data(),
                (float *)l_result);
                // 2,
                // 3 * 2,
                // 4 + l_nz_idx.size());

  std::cout<< l_nz_idx.size()<<std::endl;

  // std::cout << "\n-----------------------Result:--------------------------------\n";
  // for (int i = 0; i < 3; ++i) {
  //   for (int j = 0; j < 2; ++j) {
  //     for (int k = 0; k < 2; ++k) {
  //       std::cout << l_result[i][j][k] << "   ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }


}