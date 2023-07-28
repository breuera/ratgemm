#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <libxsmm.h>

void pad_rows(const std::vector<libxsmm_bfloat16>& i_vec_1, const std::vector<libxsmm_bfloat16>& i_vec_2, const std::vector<int64_t>& i_indices, std::vector<libxsmm_bfloat16>& o_mat_padded, const int64_t i_m, const int64_t i_n) {
  // Copy mat
  for (int64_t l_n = 0; l_n < i_n; l_n++) {
    for (int64_t l_m = 0; l_m < i_m; l_m++) {
      o_mat_padded.push_back(i_vec_1[l_m + i_m * l_n]);
    }
  }
  // Copy copies
  for (std::size_t l_co_in = 0; l_co_in < i_indices.size(); l_co_in++) {
    for (int64_t l_n = 0; l_n < i_n; l_n++) {
      int64_t l_id_in = i_n * i_indices[l_co_in] + l_n; // Calculate the new index for mat 1
      std::cout << l_id_in << std::endl;
      o_mat_padded.push_back(i_vec_2[l_id_in]);
    }
  }
}

void printAsMatrix(const std::vector<libxsmm_bfloat16>& vec, int i_m, int i_n) {
    // if (vec.size() != i_m * i_n) {
    //     std::cout << "Error: The size of the vector does not match the specified matrix dimensions.\n";
    //     return;
    // }

    for (int i = 0; i < i_m; ++i) {
        for (int j = 0; j < i_n; ++j) {
            std::cout << vec[i * i_n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
  int i_m = 35;
  int i_n = 7;

  libxsmm_bfloat16 l_matrix_1[i_m][i_n] = { 0 };

  for( int64_t l_qt = 0; l_qt < i_m; l_qt++ ) {
    for( int64_t l_md = 0; l_md < i_n; l_md++ ) {
      l_matrix_1[l_qt][l_md] = (libxsmm_bfloat16) (rand())/500;
    }
  }

  std::vector<libxsmm_bfloat16> l_vec_1(l_matrix_1[0], l_matrix_1[0] + i_m * i_n);

  // Get the indices of non-zero columns
  std::vector<int64_t> l_nonZeroIndices = {2,5};

  // for (int64_t idx : l_nonZeroIndices) {
  //   std::cout << "Row " << idx << " is non-zero." << std::endl;
  // }
  std::vector<libxsmm_bfloat16> l_mat_padded;

  pad_rows(l_vec_1, l_vec_1, l_nonZeroIndices, l_mat_padded, i_m, i_n);

  printAsMatrix(l_mat_padded, i_m + l_nonZeroIndices.size() , i_n );

  return 0;
}
