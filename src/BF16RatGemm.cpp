#include "BF16RatGemm.h"

std::vector<int64_t> rat_gemm::BF16RatGemm::col_nz(const libxsmm_bfloat16* i_mat, int i_m, int i_n){
  std::vector<libxsmm_bfloat16> l_vec(i_mat, i_mat + i_m * i_n);
  std::vector<int64_t> l_nz_idx;
  for (int64_t j = 0; j < i_n; ++j) {
    for (int64_t i = 0; i < i_m; ++i) {
      if (l_vec[i * i_n + j] != 0) {
        l_nz_idx.push_back(j);
        std::cout << "col " << j << " is non-zero." << std::endl;
        break;
      }
    }
  }
  
  return l_nz_idx;
}

libxsmm_bfloat16* rat_gemm::BF16RatGemm::vec_to_mat(const std::vector<libxsmm_bfloat16>& i_vec, int i_m, int i_n) {
  libxsmm_bfloat16* l_mat = new libxsmm_bfloat16[i_m * i_n];
  for (int64_t i = 0; i < i_m; ++i) {
    for (int64_t j = 0; j < i_n; ++j) {
      l_mat[i * i_n + j] = i_vec[i * i_n + j];
    }
  }
  return l_mat;
}

std::vector<libxsmm_bfloat16> rat_gemm::BF16RatGemm::pad_cols(const libxsmm_bfloat16* i_mat_1, const libxsmm_bfloat16* i_mat_2, const std::vector<int64_t>& i_idx, const int i_m, const int i_n) {
  // Copy mat
  std::vector<libxsmm_bfloat16> l_mat_padded;
  std::vector<libxsmm_bfloat16> l_vec_1(i_mat_1, i_mat_1 + i_m * i_n);
  std::vector<libxsmm_bfloat16> l_vec_2(i_mat_2, i_mat_2 + i_m * i_n);

  for (int64_t l_m = 0; l_m < i_m; l_m++) {
    for (int64_t l_n = 0; l_n < i_n; l_n++) {
      l_mat_padded.push_back(l_vec_1[l_m * i_n + l_n]);
    }
    for (std::size_t l_co_in = 0; l_co_in < i_idx.size(); l_co_in++) {
      int64_t l_id_in = i_idx[l_co_in] + (l_m * i_n);
      l_mat_padded.push_back(l_vec_2[l_id_in]);
    }
  }
  return l_mat_padded;
}

void rat_gemm::BF16RatGemm::print_like_mat(const std::vector<libxsmm_bfloat16>& vec, int i_m, int i_n) {
  if (vec.size() != i_m * i_n) {
    std::cout << "Error: The size of the vector does not match the specified matrix dimensions.\n";
    return;
  }

  for (int i = 0; i < i_m; ++i) {
    for (int j = 0; j < i_n; ++j) {
      printf("0x%x ", vec[i * i_n + j]);
    }
    std::cout << std::endl;
  }
}


std::vector<libxsmm_bfloat16> rat_gemm::BF16RatGemm::pad_rows(const libxsmm_bfloat16* i_mat_1, const libxsmm_bfloat16* i_mat_2, const std::vector<int64_t>& i_idx, const int64_t i_m, const int64_t i_n) {
  std::vector<libxsmm_bfloat16> l_mat_padded;
  std::vector<libxsmm_bfloat16> l_vec_1(i_mat_1, i_mat_1 + i_m * i_n);
  std::vector<libxsmm_bfloat16> l_vec_2(i_mat_2, i_mat_2 + i_m * i_n);
  // Copy mat
  for (int64_t l_n = 0; l_n < i_n; l_n++) {
    for (int64_t l_m = 0; l_m < i_m; l_m++) {
      l_mat_padded.push_back(l_vec_1[l_m + i_m * l_n]);
    }
  }
  // Copy copies
  for (std::size_t l_co_in = 0; l_co_in < i_idx.size(); l_co_in++) {
    for (int64_t l_n = 0; l_n < i_n; l_n++) {
      int64_t l_id_in = i_n * i_idx[l_co_in] + l_n; // Calculate the new index for mat 1
      l_mat_padded.push_back(l_vec_2[l_id_in]);
    }
  }
  return l_mat_padded;
}