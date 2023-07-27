#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <libxsmm.h>


std::vector<int64_t> ColNonZero(const std::vector<libxsmm_bfloat16>& vec_2nd, int i_m, int i_n){
  std::vector<int64_t> nonZeroIndices;

  for (int j = 0; j < i_n; ++j) {
      for (int i = 0; i < i_m; ++i) {
          if (vec_2nd[i * i_n + j] != 0) {
              nonZeroIndices.push_back(j);
              break;
          }
      }
  }

  return nonZeroIndices;
}

libxsmm_bfloat16* vectorToMatrix(const std::vector<libxsmm_bfloat16>& i_vec, int i_m, int i_n) {
    libxsmm_bfloat16* o_mat = new libxsmm_bfloat16[i_m * i_n];

    for (int i = 0; i < i_m; ++i) {
        for (int j = 0; j < i_n; ++j) {
            o_mat[i * i_n + j] = i_vec[i * i_n + j];
        }
    }

    return o_mat;
}

// Function to attach non-zero columns to the end of the matrix
void pad_cols(const std::vector<libxsmm_bfloat16>& io_vec_1, const std::vector<libxsmm_bfloat16>& i_vec_2, const std::vector<int64_t>& i_indices, std::vector<libxsmm_bfloat16>& o_mat_padded, const int64_t i_m, const int64_t i_n) {
  // Copy mat 0
  for (int64_t l_m = 0; l_m < i_m; l_m++) {
    for (int64_t l_n = 0; l_n < i_n; l_n++) {
      o_mat_padded.push_back(io_vec_1[l_m * i_n + l_n]);
    }
    for (std::size_t l_co_in = 0; l_co_in < i_indices.size(); l_co_in++) {
      int64_t l_id_in = i_indices[l_co_in] + (l_m * i_n);
        std::cout << l_id_in << std::endl;
      o_mat_padded.push_back(i_vec_2[l_id_in]);
    }
  }
}

void printAsMatrix(const std::vector<libxsmm_bfloat16>& vec, int i_m, int i_n) {
    if (vec.size() != i_m * i_n) {
        std::cout << "Error: The size of the vector does not match the specified matrix dimensions.\n";
        return;
    }

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

  libxsmm_bfloat16 l_matrix_2[i_m][i_n] = {{0, 0, 1, 2, 0, 4, 5},
                                            {0, 0, 0, 0, 0, 0, 0},
                                            {0, 0, 7, 8, 0, 10, 11},
                                            {0, 0, 0, 0, 0, 0, 0},
                                            {0, 0, 13, 14, 0, 16, 17}};

  std::vector<libxsmm_bfloat16> l_vec_1(l_matrix_1[0], l_matrix_1[0] + i_m * i_n);
  std::vector<libxsmm_bfloat16> l_vec_2(l_matrix_2[0], l_matrix_2[0] + i_m * i_n);

  // Get the indices of non-zero columns
  std::vector<int64_t> l_nonZeroIndices = ColNonZero(l_vec_2, i_m, i_n);

  for (int64_t idx : l_nonZeroIndices) {
    std::cout << "Column " << idx << " is non-zero." << std::endl;
  }
  std::vector<libxsmm_bfloat16> l_mat_padded;


  pad_cols(l_vec_1, l_vec_2, l_nonZeroIndices, l_mat_padded, i_m, i_n);


  printAsMatrix(l_mat_padded, i_m, (i_n + l_nonZeroIndices.size()));


  return 0;
}
