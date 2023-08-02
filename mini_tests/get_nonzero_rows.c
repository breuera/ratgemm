#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <libxsmm.h>

std::vector<int64_t> RowNonZero(const std::vector<libxsmm_bfloat16>& i_vec, int i_m, int i_n){
    std::vector<int64_t> nonZeroIndices;

    for (int i = 0; i < i_m; ++i) {
        bool rowHasNonZero = false;
        for (int j = 0; j < i_n; ++j) {
            if (i_vec[i * i_n + j] != 0) {
                rowHasNonZero = true;
                break;
            }
        }
        if (rowHasNonZero) {
            nonZeroIndices.push_back(i);
        }
    }
    return nonZeroIndices;
}

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
  int i_m = 5;
  int i_n = 7;

  libxsmm_bfloat16 l_matrix_1[i_m][i_n] = {{0, 0, 1, 2, 0, 4, 5},
                                            {0, 2, 0, 0, 0, 0, 0},
                                            {0, 0, 7, 8, 7, 10, 11},
                                            {5, 0, 0, 0, 0, 0, 0},
                                            {0, 0, 13, 14, 0, 16, 17}};

  libxsmm_bfloat16 l_matrix_2[i_m][i_n] = {{0, 0, 1, 2, 0, 4, 13},
                                            {0, 0, 0, 0, 0, 0, 0},
                                            {0, 0, 7, 8, 0, 10, 13},
                                            {0, 0, 0, 0, 0, 0, 0},
                                            {0, 0, 13, 14, 0, 16, 13}};

  std::vector<libxsmm_bfloat16> l_vec_1(l_matrix_1[0], l_matrix_1[0] + i_m * i_n);
  std::vector<libxsmm_bfloat16> l_vec_2(l_matrix_2[0], l_matrix_2[0] + i_m * i_n);

  // Get the indices of non-zero columns
  std::vector<int64_t> l_nonZeroIndices = RowNonZero(l_vec_2, i_m, i_n);

  // for (int64_t idx : l_nonZeroIndices) {
  //   std::cout << "Row " << idx << " is non-zero." << std::endl;
  // }
  std::vector<libxsmm_bfloat16> l_mat_padded;

  pad_rows(l_vec_1, l_vec_2, l_nonZeroIndices, l_mat_padded, i_m, i_n);

  printAsMatrix(l_mat_padded, i_m + l_nonZeroIndices.size() , i_n );

  return 0;
}
