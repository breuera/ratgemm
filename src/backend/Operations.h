#ifndef RAT_GEMM_BACKEND_OPERATIONS_H
#define RAT_GEMM_BACKEND_OPERATIONS_H

#include <cstdint>
#include <libxsmm.h>

namespace rat_gemm {
  namespace backend {
    class Operations;
  }
}

class rat_gemm::backend::Operations {
  private:
    libxsmm_meltwfunction_unary m_gather_rows;

  public:
    /**
     * Initializes an operation which gathers rows from A and stores them in B.
     *
     * @param i_m number of rows which are gathered from A.
     * @param i_n number of columns in A and B.
     * @param i_ld_in leading dimension of A.
     * @param i_ld_out leading dimension of B.
     **/
    void init_gather_rows( int64_t i_m,
                           int64_t i_n,
                           int64_t i_ld_in,
                           int64_t i_ld_out );

    /**
     * Applies the operation which gathers rows from A (mat 0) and stores them to B (mat 1).
     *
     * @param i_row_ids ids of the rows.
     * @param i_mat_0 matrix A from which the rows are gathered.
     * @param i_mat_1 matrix B to which the rows are written. 
     **/
    void apply_gather_rows( int64_t * i_row_ids,
                            float   * i_mat_0,
                            float   * i_mat_1 );
};

#endif
