#include "RationalMatrix.h"

std::vector< float > rat_gemm::backend::RationalMatrix::mask( int64_t              i_m,
                                                              int64_t              i_n,
                                                              int64_t              i_ld_in,
                                                              int64_t              i_ld_out,
                                                              uint32_t             i_mask,
                                                              std::vector< float > i_mat ) {
  std::vector< float > l_mat_masked( i_n * i_ld_out );

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      int64_t l_id_in  = l_n * i_ld_in  + l_m;
      int64_t l_id_out = l_n * i_ld_out + l_m;

      float l_val_fp32 = i_mat[l_id_in];
      float l_val_masked = l_val_fp32;
      uint32_t * l_val_masked_ptr = (uint32_t*) &l_val_masked;
      *l_val_masked_ptr &= i_mask;

      l_mat_masked[l_id_out] = l_val_masked;
    }
  }

  return l_mat_masked;
}

std::vector< float > rat_gemm::backend::RationalMatrix::diff( int64_t              i_m,
                                                              int64_t              i_n,
                                                              int64_t              i_ld_in_0,
                                                              int64_t              i_ld_in_1,
                                                              int64_t              i_ld_out,
                                                              std::vector< float > i_mat_0,
                                                              std::vector< float > i_mat_1 ) {
  std::vector< float > l_diff( i_n * i_ld_out );

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      int64_t l_id_in_0 = l_n * i_ld_in_0 + l_m;
      int64_t l_id_in_1 = l_n * i_ld_in_1 + l_m;
      int64_t l_id_out  = l_n * i_ld_out  + l_m;

      l_diff[l_id_out] = i_mat_0[l_id_in_0] - i_mat_1[l_id_in_1];
    }
  }

  return l_diff;
}


std::vector< int64_t > rat_gemm::backend::RationalMatrix::nz_rows( int64_t              i_m,
                                                                   int64_t              i_n,
                                                                   int64_t              i_ld,
                                                                   std::vector< float > i_mat ) {
  std::vector< int64_t > l_nz_rows;

  for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
    for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
      int64_t l_id  = l_n * i_ld + l_m;
      
      if( i_mat[l_id] != 0.0 ) {
        l_nz_rows.push_back( l_m );
        break;
      }
    }
  }

  return l_nz_rows;
}

std::vector< int64_t > rat_gemm::backend::RationalMatrix::nz_cols( int64_t              i_m,
                                                                   int64_t              i_n,
                                                                   int64_t              i_ld,
                                                                   std::vector< float > i_mat ) {
  std::vector< int64_t > l_nz_cols;

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      int64_t l_id  = l_n * i_ld + l_m;
      
      if( i_mat[l_id] != 0.0 ) {
        l_nz_cols.push_back( l_n );
        break;
      }
    }
  }

  return l_nz_cols;
}

std::vector< float > rat_gemm::backend::RationalMatrix::pad_cols( int64_t                i_m,
                                                                  int64_t                i_n_in_0,
                                                                  int64_t                i_ld_in_0,
                                                                  int64_t                i_ld_in_1,
                                                                  int64_t                i_ld_out,
                                                                  std::vector< int64_t > l_col_ids,
                                                                  std::vector< float >   i_mat_0,
                                                                  std::vector< float >   i_mat_1 ) {
  int64_t l_size_out  = i_n_in_0 * i_ld_out;
          l_size_out += l_col_ids.size() * i_ld_out;
  std::vector< float > l_mat_padded( l_size_out );

  // copy mat 0
  for( int64_t l_n = 0; l_n < i_n_in_0; l_n++ ) {
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      int64_t l_id_in  = l_n * i_ld_in_0 + l_m;
      int64_t l_id_out = l_n * i_ld_out  + l_m; 

      l_mat_padded[l_id_out] = i_mat_0[l_id_in];
    }
  }

  // pad columns
  for( std::size_t l_co = 0; l_co < l_col_ids.size(); l_co++ ) {
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      int64_t l_id_in = l_col_ids[l_co] * i_ld_in_1 + l_m;
      int64_t l_id_out = (i_n_in_0+l_co) * i_ld_out  + l_m;

      l_mat_padded[l_id_out] = i_mat_1[l_id_in];
    }
  }

  return l_mat_padded;
}