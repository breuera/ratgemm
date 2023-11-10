#include "BF16RatGemm.h"

std::vector<int64_t> rat_gemm::BF16RatGemm::col_nz(const libxsmm_bfloat16* i_mat, int i_m, int i_n){

  std::vector<libxsmm_bfloat16> l_vec(i_mat, i_mat + i_m * i_n);
  std::vector<int64_t> l_nz_idx;

  for (int64_t l_n = 0; l_n < i_n; ++l_n) {
    for (int64_t l_m = 0; l_m < i_m; ++l_m) {
      if (l_vec[l_m * i_n + l_n] != 0) {
        l_nz_idx.push_back(l_n);
        std::cout << "col " << l_n << " is non-zero." << std::endl;
        break;
      }
    }
  }
  
  return l_nz_idx;
}

std::vector<int64_t> rat_gemm::BF16RatGemm::row_nz(const libxsmm_bfloat16* i_mat, int i_m, int i_n){

    std::vector<libxsmm_bfloat16> l_vec(i_mat, i_mat + i_m * i_n);
    std::vector<int64_t> l_nz_idx;

    for (int64_t l_m = 0; l_m < i_m; ++l_m) {
        bool l_nz_tmp = false;
        for (int64_t l_n = 0; l_n < i_n; ++l_n) {
            if (l_vec[l_m * i_n + l_n] != 0) {
                l_nz_tmp = true;
                break;
            }
        }
        if (l_nz_tmp) {
            l_nz_idx.push_back(l_m);
        }
    }
    return l_nz_idx;
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

std::vector<libxsmm_bfloat16> rat_gemm::BF16RatGemm::pad_rows(const libxsmm_bfloat16* i_mat_1, const libxsmm_bfloat16* i_mat_2, const std::vector<int64_t>& i_idx, const int64_t i_m, const int64_t i_n){
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
      int64_t l_id_in = i_n * i_idx[l_co_in] + l_n;
      l_mat_padded.push_back(l_vec_2[l_id_in]);
    }
  }
  return l_mat_padded;
}

void rat_gemm::BF16RatGemm::vec_print_like_mat(const std::vector<libxsmm_bfloat16>& i_vec, int i_m, int i_n) {
  if (i_vec.size() != i_m * i_n) {
    std::cout << "Error: The size of the vector does not match the specified matrix dimensions.\n";
    return;
  }
  int l_spacing = 4;

  for (int l_m = 0; l_m < i_m; ++l_m) {
    for (int l_n = 0; l_n < i_n; ++l_n) {
      std::cout << "0x" << std::hex << std::setw(l_spacing) << static_cast<int>(i_vec[l_m * i_n + l_n]) << " ";
    }
    std::cout << std::endl;
  }
}

void rat_gemm::BF16RatGemm::print_mat(const libxsmm_bfloat16* i_mat, int i_m, int i_n){
  // Assuming that i_mat is a row-major matrix of size i_m x i_n

  // Set the precision for printing floating-point values
  std::cout << std::fixed << std::setprecision(6);

  for (int row = 0; row < i_m; ++row) {
    for (int col = 0; col < i_n; ++col) {
      // Calculate the index of the element (row, col) in the 1D array
      int index = row * i_n + col;

      // Access the element using the calculated index and cast it to float for printing
      float value = static_cast<float>(i_mat[index]);

      // Print the element value with a tab separator
      std::cout << "0x" << std::hex << static_cast<int>(value) << "\t";
    }
    // Move to the next row after printing all elements in the current row
    std::cout << std::endl;
  }
}

float* rat_gemm::BF16RatGemm::bf16_gemm(const libxsmm_bfloat16* i_dof_1,
                                        const libxsmm_bfloat16* i_dof_2,
                                        libxsmm_bfloat16* i_stiff,
                                        int i_m,
                                        int i_k,
                                        int i_n ){ // m*n   n*k
/* ------------------------------------------------------------------------------------------ */
  // Temporary buffer to hold the result of (A + B)

  // float* l_tmp = new float[i_m * i_k];
  // float* l_result = new float[i_m * i_n];
  // libxsmm_float_uint hybrid_in = {0};

  // // Perform element-wise addition of A and B
  // for (int i = 0; i < i_m; ++i) {
  //   for (int j = 0; j < i_k; ++j) {
  //     l_tmp[i * i_k + j] = i_dof_1[i * i_k + j] + i_dof_2[i * i_k + j];
  //     // std::cout << "0x" << std::hex << static_cast<int>(l_tmp[i * i_k + j]) << "=";
  //     // std::cout << "0x" << std::hex << static_cast<int>(i_dof_1[i * i_k + j]) << "+";
  //     // std::cout << "0x" << std::hex << static_cast<int>(i_dof_2[i * i_k + j] ) << std::endl;
  //   }
  // }

  // for (int i = 0; i < i_m; ++i) {
  //   for (int j = 0; j < i_n; ++j) {
  //     float l_sum = static_cast<float>(0.0);
  //     for (int k = 0; k < i_k; ++k) {
  //       // Indexing formulas for accessing elements in l_tmp and i_stiff matrices
  //       hybrid_in.f =(i_dof_1[i * i_k + k] * i_stiff[k * i_n + j]);
  //       // Create a pointer to the unsigned int value
  //       unsigned int* uintPointer = &(hybrid_in.u);
  //       // Create a float pointer and reinterpret the bits from the uint pointer
  //       float* floatPointer = reinterpret_cast<float*>(uintPointer);
  //       // Access the float value without changing the bits
  //       float floatValue = *floatPointer;
  //       l_sum += floatValue;
  //       // printf("0x%x ", hybrid_in.u);
  //     }
  //     // std::cout << std::hex << l_tmp[i * i_k] <<" * "<< i_stiff[j * i_n] << "= ";
  //     // std::cout << l_sum << std::endl;
  //     l_result[i * i_n + j] = l_sum;
  //   }
  // }

  float* l_result = new float[i_m * i_n];

  for (int i = 0; i < i_m; ++i) {
    for (int j = 0; j < i_n; ++j) {
      float l_sum = static_cast<float>(0.0);
      float l_sum_2 = static_cast<float>(0.0);
      for (int k = 0; k < i_k; ++k) {
        // Indexing formulas for accessing elements in l_tmp and i_stiff matrices
        l_sum =(i_dof_1[i * i_k + k] * i_stiff[k * i_n + j]);
        l_sum_2 =(i_dof_2[i * i_k + k] * i_stiff[k * i_n + j]);
      }
      
      // std::cout << std::hex << l_tmp[i * i_k] <<" * "<< i_stiff[j * i_n] << "= ";
      // std::cout << l_sum << std::endl;
      l_result[i * i_n + j] = l_sum + l_sum_2;
    }
  }
  
  
/* ------------------------------------------------------------------------------------------ */
  //  libxsmm_gemmfunction m_sgemm = nullptr;
  // Perform matrix multiplication (A + B) * D using libxsmm_gemm
  // libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  //                  l_flags |= LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI;
  //                  l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  // libxsmm_bitfield l_prefetch_flags = 0;
  // const libxsmm_blasint lda = i_m, ldb = i_n, ldc = i_k;

  // libxsmm_gemm_shape l_shape = libxsmm_create_gemm_shape( i_m,
  //                                                         i_n,
  //                                                         i_k,
  //                                                         lda,
  //                                                         ldb,
  //                                                         ldc,
  //                                                         LIBXSMM_DATATYPE_F32,
  //                                                         LIBXSMM_DATATYPE_F32,
  //                                                         LIBXSMM_DATATYPE_F32,
  //                                                         LIBXSMM_DATATYPE_F32 );
  // libxsmm_gemm_param l_param;
  // memset( &l_param,
  //         0,
  //         sizeof(libxsmm_gemm_param) );
  // l_param.a.primary = l_tmp;
  // l_param.b.primary = i_stiff;
  // l_param.c.primary = o_result;

  // m_sgemm = libxsmm_dispatch_gemm_v2( l_shape,
  //                                     l_flags,
  //                                     l_prefetch_flags );

  // m_sgemm( &l_param );

/* ------------------------------------------------------------------------------------------ */
//   libxsmm_bfloat16* l_result_1 = new libxsmm_bfloat16[i_m * i_n];
//   libxsmm_bfloat16* l_result_2 = new libxsmm_bfloat16[i_m * i_n];
//   for (int i = 0; i < i_m; ++i) {
//     for (int j = 0; j < i_n; ++j) {
//       libxsmm_bfloat16 l_sum = static_cast<libxsmm_bfloat16>(0.0);
//       for (int k = 0; k < i_k; ++k) {
//         // Indexing formulas for accessing elements in l_tmp and i_stiff matrices
//         l_sum += l_tmp[i * i_k + k] * i_stiff[k * i_n + j];
//       }
//       // std::cout << std::hex << l_tmp[i * i_k] <<" * "<< i_stiff[j * i_n] << "= ";
//       // std::cout << l_sum << std::endl;
//       l_result[i * i_n + j] = l_sum;
//     }
//   }

//   // Free the temporary buffer
  // delete(l_tmp);
  return l_result;
}

