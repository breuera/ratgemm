#ifndef BF16_RAT_GEMM_RAT_GEMM_H
#define BF16_RAT_GEMM_RAT_GEMM_H

#include <cstdint>
#include <vector>
#include <libxsmm.h>
#include <iostream>
#include <iomanip>

namespace rat_gemm {
  class BF16RatGemm;
}

class rat_gemm::BF16RatGemm {
  public:

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    std::vector<int64_t> col_nz(const libxsmm_bfloat16* i_mat, int i_m, int i_n);

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    std::vector<int64_t> row_nz(const libxsmm_bfloat16* i_mat, int i_m, int i_n);

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    std::vector<libxsmm_bfloat16> pad_cols(const libxsmm_bfloat16* i_mat_1, const libxsmm_bfloat16* i_mat_2, const std::vector<int64_t>& i_idx, const int i_m, const int i_n);

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    std::vector<libxsmm_bfloat16> pad_rows(const libxsmm_bfloat16* i_mat_1, const libxsmm_bfloat16* i_mat_2, const std::vector<int64_t>& i_idx, const int64_t i_m, const int64_t i_n);

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    void vec_print_like_mat(const std::vector<libxsmm_bfloat16>& i_vec, int i_m, int i_n);

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    void print_mat(const libxsmm_bfloat16* i_mat, int i_m, int i_n);

    /**
     * Computes C=A*B for the rational matrix A.
     * 
     * @param i_b matrix B.
     * @param o_c matrix C.
     */
    float* bf16_gemm(const libxsmm_bfloat16* i_dof_1,
                   const libxsmm_bfloat16* i_dof_2,
                   libxsmm_bfloat16* i_stiff,
                   int i_m,
                   int i_n,
                   int i_k);

};

#endif
