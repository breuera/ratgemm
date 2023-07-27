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


void pad_rows(const std::vector<libxsmm_bfloat16>& io_vec_1, const std::vector<libxsmm_bfloat16>& i_vec_2, const std::vector<int64_t>& i_indices, std::vector<libxsmm_bfloat16>& o_vec_padded, const int64_t i_m, const int64_t i_n) {
  // Copy mat
  for (int64_t l_n = 0; l_n < i_n; l_n++) {
    for (int64_t l_m = 0; l_m < i_m; l_m++) {
      o_vec_padded.push_back(io_vec_1[l_m + i_m * l_n]);
    }
  }
  // Copy copies
  for (std::size_t l_co_in = 0; l_co_in < i_indices.size(); l_co_in++) {
    for (int64_t l_n = 0; l_n < i_n; l_n++) {
      int64_t l_id_in = i_n * i_indices[l_co_in] + l_n; // Calculate the new index for mat 1
      std::cout << l_id_in << std::endl;
      o_vec_padded.push_back(i_vec_2[l_id_in]);
    }
  }
}