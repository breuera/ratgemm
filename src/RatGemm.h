#ifndef RAT_GEMM_RAT_GEMM_H
#define RAT_GEMM_RAT_GEMM_H

#include <cstdint>
#include <vector>
#include "backend/Operations.h"

namespace rat_gemm {
  class RatGemm;
}

class rat_gemm::RatGemm {
  private:
    backend::Operations    m_ops;
    std::vector< float >   m_rat_a;
    std::vector< float >   m_scratch;
    std::vector< int64_t > m_gather_row_ids;

    int64_t  m_n = 0;
    int64_t  m_k = 0;
    int64_t  m_k_padded = 0;
    int64_t  m_ld_b = 0;
    uint32_t m_mask = 0xffff0000;

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
    void apply( float * i_b,
                float * o_c );
};

#endif
