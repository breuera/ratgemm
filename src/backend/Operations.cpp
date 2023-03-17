#include <cassert>
#include "Operations.h"

void rat_gemm::backend::Operations::init_gather_rows( int64_t i_m,
                                                      int64_t i_n,
                                                      int64_t i_ld_in,
                                                      int64_t i_ld_out ) {
  libxsmm_meltw_unary_flags l_flags = LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS;
  l_flags = LIBXSMM_EOR( libxsmm_meltw_unary_flags,
                         l_flags,
                         LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES );

  libxsmm_meltw_unary_type l_type = LIBXSMM_MELTW_TYPE_UNARY_GATHER;

  libxsmm_datatype l_dtype = LIBXSMM_DATATYPE_F32;

  libxsmm_meltw_unary_shape l_shape = libxsmm_create_meltw_unary_shape( i_m,
                                                                        i_n,
                                                                        i_ld_in,
                                                                        i_ld_out,
                                                                        l_dtype,
                                                                        l_dtype,
                                                                        l_dtype );

  m_gather_rows = libxsmm_dispatch_meltw_unary_v2( l_type,
                                                   l_shape,
                                                   l_flags );

  assert( m_gather_rows != NULL );
}

void rat_gemm::backend::Operations::apply_gather_rows( int64_t * i_row_ids,
                                                       float   * i_mat_0,
                                                       float   * i_mat_1 ) {
  libxsmm_meltw_unary_param l_param;
  memset( &l_param,
          0,
          sizeof( l_param ) );

  l_param.in.primary = i_mat_0;
  l_param.in.secondary = i_row_ids;
  l_param.out.primary = i_mat_1;

  assert( m_gather_rows != NULL );
  m_gather_rows( &l_param );
}