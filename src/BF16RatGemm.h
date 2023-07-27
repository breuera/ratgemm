#ifndef NEW_RAT_GEMM_RAT_GEMM_H
#define NEW_RAT_GEMM_RAT_GEMM_H

#include <cstdint>
#include <vector>
#include <libxsmm.h>

namespace rat_gemm {
  class RatGemm;
}

class rat_gemm::RatGemm {

  public:
    /**
     * Initializes rational GEMM for C=A*B,
     * where A is a (pre-processed) rational matrix.
     * 
     * @param i_m BLAS parameter m.
     * @param i_n BLAS parameter n.
     * @param i_k BLAS parameter k.
     * @param i_ld_a leading dimension of A.
     * @param i_ld_b leading dimension of B.
     * @param i_ld_c leading dimension of C.
     * @param i_rat_a rational matrix A.
     **/
    void init( int64_t i_m,
               int64_t i_n,
               int64_t i_k,
               int64_t i_ld_a,
               int64_t i_ld_b,
               int64_t i_ld_c,
               float * i_rat_a );

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    std::vector<int64_t> ColNonZero(const std::vector<libxsmm_bfloat16>& vec_2nd, int i_m, int i_n);

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    libxsmm_bfloat16* vectorToMatrix(const std::vector<libxsmm_bfloat16>& i_vec, int i_m, int i_n);

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    void pad_cols(const std::vector<libxsmm_bfloat16>& io_vec_1, const std::vector<libxsmm_bfloat16>& i_vec_2, const std::vector<int64_t>& i_indices, std::vector<libxsmm_bfloat16>& o_mat_padded, const int64_t i_m, const int64_t i_n);

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    void printAsMatrix(const std::vector<libxsmm_bfloat16>& vec, int i_m, int i_n);
  
    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    void pad_rows(const std::vector<libxsmm_bfloat16>& io_vec_1, const std::vector<libxsmm_bfloat16>& i_vec_2, const std::vector<int64_t>& i_indices, std::vector<libxsmm_bfloat16>& o_vec_padded, const int64_t i_m, const int64_t i_n)
};

#endif
