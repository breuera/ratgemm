#include "RatGemm.h"
#include "backend/RationalMatrix.h"

void rat_gemm::RatGemm::init( int64_t i_m,
                              int64_t i_n,
                              int64_t i_k,
                              int64_t i_ld_a,
                              int64_t i_ld_b,
                              int64_t i_ld_c,
                              float * i_rat_a ) {
  std::vector< float > l_mat( i_rat_a,
                              i_rat_a+(i_k*i_ld_a) );

  std::vector< float > l_mat_masked = backend::RationalMatrix::mask( i_m,
                                                                     i_k,
                                                                     i_ld_a,
                                                                     i_m,
                                                                     m_mask,
                                                                     l_mat  );

  std::vector< float > l_mat_err = backend::RationalMatrix::diff( i_m,
                                                                  i_k,
                                                                  i_ld_a,
                                                                  i_m,
                                                                  i_m,
                                                                  l_mat,
                                                                  l_mat_masked );

  std::vector< int64_t > l_nz_cols = backend::RationalMatrix::nz_cols( i_m,
                                                                       i_k,
                                                                       i_m,
                                                                       l_mat_err );

  std::vector< float > l_mat_padded = backend::RationalMatrix::pad_cols( i_m,
                                                                         i_k,
                                                                         i_m,
                                                                         i_m,
                                                                         i_m,
                                                                         l_nz_cols,
                                                                         l_mat_masked,
                                                                         l_mat_err );

  // TOOD: add check that matrix is represented exactly

  int64_t l_k_plus = l_nz_cols.size();
  int64_t l_k_padded = i_k + l_k_plus;

  m_ops.init_gather_rows( l_k_plus,
                          i_n,
                          i_ld_b,
                          l_k_padded );

  m_ops.init_sgemm( i_m,
                    i_n,
                    l_k_padded,
                    i_m,
                    l_k_padded,
                    i_ld_c );

  m_gather_row_ids = l_nz_cols;
  m_rat_a = l_mat_padded;
  m_scratch.resize( i_m * l_k_padded );

  m_n = i_n;
  m_k = i_k;
  m_k_padded = l_k_padded;
  m_ld_b = i_ld_b;
}

void rat_gemm::RatGemm::apply( float * i_b,
                               float * o_c ) {
  for( int64_t l_n = 0; l_n < m_n; l_n++ ) {
    for( int64_t l_k = 0; l_k < m_k; l_k++ ) {
      int64_t l_id_in = l_n * m_ld_b + l_k;
      int64_t l_id_out = l_n * m_k_padded + l_k;

      m_scratch[l_id_out] = i_b[l_id_in];
    }
  }

  int64_t l_off = m_k;
  m_ops.apply_gather_rows( m_gather_row_ids.data(),
                           i_b,
                           m_scratch.data()+l_off );

  m_ops.apply_sgemm( m_rat_a.data(),
                     m_scratch.data(),
                     o_c );
}