#ifndef RAT_GEMM_BACKEND_RATIONAL_MATRIX_H
#define RAT_GEMM_BACKEND_RATIONAL_MATRIX_H

#include <cstdint>
#include <vector>

namespace rat_gemm {
  namespace backend {
    class RationalMatrix;
  }
}

class rat_gemm::backend::RationalMatrix {
  private:
  public:
    /**
     * Element-wise applies a mask to a column-major matrix.
     *
     * @param i_m number of rows. 
     * @param i_n number of columns.
     * @param i_ld_in leading dimension of input matrix.
     * @param i_ld_out leading dimension of output matrix.
     * @param i_mask mask.
     * @param i_mat input matrix.
     * @return masked matrix
     **/
    static std::vector< float > mask( int64_t              i_m,
                                      int64_t              i_n,
                                      int64_t              i_ld_in,
                                      int64_t              i_ld_out,
                                      uint32_t             i_mask,
                                      std::vector< float > i_mat );

    /**
     * Computes the difference C=A-B of two matrices A and B.
     *
     * @param i_m number of rows.
     * @param i_n number of columns
     * @param i_ld_in_0 leading dimension of the first input (A).
     * @param i_ld_in_1 leading dimension of the second input (B).
     * @param i_ld_out leading dimension of the output (C).
     * @param i_mat_0 first input matrix (A).
     * @param i_mat_1 second input matrix (B).
     * @return difference (C).
     **/
    static std::vector< float > diff( int64_t              i_m,
                                      int64_t              i_n,
                                      int64_t              i_ld_in_0,
                                      int64_t              i_ld_in_1,
                                      int64_t              i_ld_out,
                                      std::vector< float > i_mat_0,
                                      std::vector< float > i_mat_1 );

    /**
     * Derives the ids of non-zero rows in a matrix.
     *
     * @param i_m number of rows.
     * @param i_n number of columns.
     * @param i_ld leading dimensions.
     * @param i_mat matrix.
     * @return ids of non-zero rows.
     **/
    static std::vector< int64_t > nz_rows( int64_t              i_m,
                                           int64_t              i_n,
                                           int64_t              i_ld,
                                           std::vector< float > i_mat );

    /**
     * Derives the ids of non-zero columns in a matrix.
     *
     * @param i_m number of rows.
     * @param i_n number of columns.
     * @param i_ld leading dimensions.
     * @param i_mat matrix.
     * @return ids of non-zero columns.
     **/
    static std::vector< int64_t > nz_cols( int64_t              i_m,
                                           int64_t              i_n,
                                           int64_t              i_ld,
                                           std::vector< float > i_mat );

    /**
     * Uses columns of a matrix B to pads them to a matrix A.
     *
     * @param i_m number of rows.
     * @param i_n_in_0 number of rows of first matrix (A).
     * @param i_ld_in_0 leading dimension of first matrix (A).
     * @param i_ld_in_1 leading dimensions of second matrix (B).
     * @param i_ld_out leading dimensions of padded output matrix.
     * @param i_col_ids ids of the columns which are padded.
     * @param i_mat_0 first matrix (A).
     * @param i_mat_1 second matrix (B).
     * @return padded output matrix.
     */
    static std::vector< float > pad_cols( int64_t                i_m,
                                          int64_t                i_n_in_0,
                                          int64_t                i_ld_in_0,
                                          int64_t                i_ld_in_1,
                                          int64_t                i_ld_out,
                                          std::vector< int64_t > l_col_ids,
                                          std::vector< float >   i_mat_0,
                                          std::vector< float >   i_mat_1 );
};

#endif
