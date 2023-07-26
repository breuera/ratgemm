#ifndef RAT_GEMM_BACKEND_CONVERT_MATRIX_H
#define RAT_GEMM_BACKEND_CONVERT_MATRIX_H

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <libxsmm.h>

namespace rat_gemm {
  namespace backend {
    class ConvertMatrix;
  }
}

class rat_gemm::backend::ConvertMatrix {
  public:
    ConvertMatrix() {}
    /**
     * Subtracts two matrices A and stores them in B.
     *
     * @param io_matrix matrix A which contains random values for test.
     * @param i_s length of all matrices
     **/
    void rat_gemm::backend::ConvertMatrix::generateRandomMatrix(float * io_matrix,
                                                                int i_s);

    /**
     * Subtracts two matrices A and stores them in B.
     *
     * @param i_matrix matrix A which contains the original value.
     * @param i_matrix_h1 matrix contains the first half in BF16.
     * @param i_matrix_h2 matrix contains the second half in BF16.
     * @param o_diff matrix contains the second half in BF16.
     * @param i_s length of all matrices
     **/
    void rat_gemm::backend::ConvertMatrix::diff(float *i_matrix,
                                                const float *i_matrix_h1,
                                                const float *i_matrix_h2,
                                                float* o_diff,
                                                int i_s);
  
    /**
     * Subtracts two matrices A and stores them in B.
     *
     * @param i_matrix matrix A which contains the original value.
     * @param o_matrix_bf16_h1 matrix contains the first half in BF16.
     * @param o_matrix_bf16_h2 matrix contains the second half in BF16.
     * @param i_s length of all matrices
     **/
    void convert_fp32_two_bf16(const float* i_matrix,
                               libxsmm_bfloat16* o_matrix_bf16_h1,
                               libxsmm_bfloat16* o_matrix_bf16_h2,
                               int i_s);
};

#endif