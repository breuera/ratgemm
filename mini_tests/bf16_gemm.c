#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <iomanip> // For setting precision when printing floating-point values
#include <random> // For random number generation

#include <libxsmm.h>

void bf16_gemm(const libxsmm_bfloat16* i_dof_1,
               const libxsmm_bfloat16* i_dof_2,
               libxsmm_bfloat16* i_stiff,
               int i_m,
               int i_n,
               int i_k,
               libxsmm_bfloat16* o_result){

  libxsmm_gemmfunction m_sgemm = nullptr;

  // Temporary buffer to hold the result of (A + B)
  libxsmm_bfloat16* l_tmp = (libxsmm_bfloat16*)malloc(i_m * i_k * sizeof(libxsmm_bfloat16));

  // Perform element-wise addition of A and B
  for (int l_s = 0; l_s < i_m * i_k; ++l_s) {
    l_tmp[l_s] = i_dof_1[l_s] + i_dof_2[l_s];
  }


  // Perform matrix multiplication (A + B) * D using libxsmm_gemm
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
                   l_flags |= LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI;
                   l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  libxsmm_bitfield l_prefetch_flags = 0;
  const libxsmm_blasint lda = i_m, ldb = i_k, ldc = i_m;

  libxsmm_gemm_shape l_shape = libxsmm_create_gemm_shape( i_m,
                                                          i_n,
                                                          i_k,
                                                          lda,
                                                          ldb,
                                                          ldc,
                                                          LIBXSMM_DATATYPE_F32,
                                                          LIBXSMM_DATATYPE_F32,
                                                          LIBXSMM_DATATYPE_F32,
                                                          LIBXSMM_DATATYPE_F32 );
  libxsmm_gemm_param l_param;
  memset( &l_param,
          0,
          sizeof(libxsmm_gemm_param) );
  l_param.a.primary = l_tmp;
  l_param.b.primary = i_stiff;
  l_param.c.primary = o_result;

  m_sgemm = libxsmm_dispatch_gemm_v2( l_shape,
                                      l_flags,
                                      l_prefetch_flags );

  m_sgemm( &l_param );

  // Free the temporary buffer
  free(l_tmp);
}

void printMatrix(const libxsmm_bfloat16* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Indexing formula for a 2D matrix: element at row i, column j is matrix[i * cols + j]
            std::cout << std::fixed << std::setprecision(2) << float(matrix[i * cols + j]) << " ";
        }
        std::cout << std::endl;
    }
}

void initializeMatrixRandom(libxsmm_bfloat16* matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f); // Adjust the range based on your requirements

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float random_value = dist(gen);
            matrix[i * cols + j] = static_cast<libxsmm_bfloat16>(random_value);
        }
    }
}

int main() {
  // Test case 1: Test the function with small input matrices
  const int i_m = 2;
  const int i_n = 3;
  const int i_k = 4;
  const int size = i_m * i_k;

  // Create input matrices
  libxsmm_bfloat16* i_dof_1 = new libxsmm_bfloat16[size];
  libxsmm_bfloat16* i_dof_2 = new libxsmm_bfloat16[size];
  libxsmm_bfloat16* i_stiff = new libxsmm_bfloat16[i_k * i_n];
  libxsmm_bfloat16* o_result = new libxsmm_bfloat16[i_m * i_n];
  
  // Initialize input matrices with some values
  initializeMatrixRandom(i_dof_1, i_m, i_k);
  initializeMatrixRandom(i_dof_2, i_m, i_k);
  initializeMatrixRandom(i_stiff, i_k, i_n);

  // Call the function to be tested
  bf16_gemm(i_dof_1, i_dof_2, i_stiff, i_m, i_n, i_k, o_result);
  std::cout << "result matrix:" << std::endl;
  printMatrix(o_result, i_m, i_k);

  // Clean up
  delete[] i_dof_1;
  delete[] i_dof_2;
  delete[] i_stiff;
  delete[] o_result;
  return 0;
}

