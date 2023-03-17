#include <catch2/catch.hpp>
#include "Operations.h"

TEST_CASE( "Gather rows operation using a small FP32 example matrix.",
           "[operations_gather_rows]" ) {
  rat_gemm::backend::Operations l_ops;

  int64_t l_m = 5;
  int64_t l_n = 3;
  int64_t l_ld_in = 7;
  int64_t l_ld_out = 6;

  l_ops.init_gather_rows( l_m,
                          l_n,
                          l_ld_in,
                          l_ld_out );

  float l_a[3][7];
  float l_b[3][6] = { 0 };

  for( int64_t l_co = 0; l_co < 3; l_co++ ) {
    for( int64_t l_ro = 0; l_ro < 7; l_ro++ ) {
      l_a[l_co][l_ro] = l_co * 7 + l_ro;
    }
  }

  int64_t l_ids_row[5] = {4, 1, 2, 0, 1};
  l_ops.apply_gather_rows( l_ids_row,
                           l_a[0],
                           l_b[0] );

  REQUIRE( l_b[0][0] ==  4.0f );
  REQUIRE( l_b[1][0] == 11.0f );
  REQUIRE( l_b[2][0] == 18.0f );

  REQUIRE( l_b[0][1] ==  1.0f );
  REQUIRE( l_b[1][1] ==  8.0f );
  REQUIRE( l_b[2][1] == 15.0f );

  REQUIRE( l_b[0][2] ==  2.0f );
  REQUIRE( l_b[1][2] ==  9.0f );
  REQUIRE( l_b[2][2] == 16.0f );

  REQUIRE( l_b[0][3] ==  0.0f );
  REQUIRE( l_b[1][3] ==  7.0f );
  REQUIRE( l_b[2][3] == 14.0f );

  REQUIRE( l_b[0][4] ==  1.0f );
  REQUIRE( l_b[1][4] ==  8.0f );
  REQUIRE( l_b[2][4] == 15.0f );
}

TEST_CASE( "SGEMM operation on small example matrices.",
           "[operations_sgemm]" ) {
  int64_t l_m = 3;
  int64_t l_n = 4;
  int64_t l_k = 2;

  int64_t l_ld_a = 5;
  int64_t l_ld_b = 4;
  int64_t l_ld_c = 6;

  float l_a[2][5] = { {1, 2, 3, 4,  5},
                      {6, 7, 8, 9, 10} };

  float l_b[4][4] = {  { 1,  2,  3,  4},
                       { 5,  6,  7,  8},
                       { 9, 10, 11, 12},
                       {13, 14, 15, 16} };

  float l_c[4][6] = { 0 };

  rat_gemm::backend::Operations l_ops;

  l_ops.init_sgemm( l_m,
                    l_n,
                    l_k,
                    l_ld_a,
                    l_ld_b,
                    l_ld_c  );

  l_ops.apply_sgemm( l_a[0],
                     l_b[0],
                     l_c[0] );

  /*
   * import numpy
   * l_a = numpy.matrix( [[1, 6], [2, 7], [3, 8]] )
   * l_b = numpy.matrix( [[1, 5, 9, 13], [2, 6, 10, 14]] )
   * l_a @ l_b
   */
  REQUIRE( l_c[0][0] == Approx( 13) );
  REQUIRE( l_c[1][0] == Approx( 41) );
  REQUIRE( l_c[2][0] == Approx( 69) );
  REQUIRE( l_c[3][0] == Approx( 97) );

  REQUIRE( l_c[0][1] == Approx( 16) );
  REQUIRE( l_c[1][1] == Approx( 52) );
  REQUIRE( l_c[2][1] == Approx( 88) );
  REQUIRE( l_c[3][1] == Approx(124) );

  REQUIRE( l_c[0][2] == Approx( 19) );
  REQUIRE( l_c[1][2] == Approx( 63) );
  REQUIRE( l_c[2][2] == Approx(107) );
  REQUIRE( l_c[3][2] == Approx(151) );

  for( int64_t l_co = 0; l_co < 4; l_co++ ) {
    for( int64_t l_ro = 3; l_ro < 6; l_ro++ ) {
      REQUIRE( l_c[l_co][l_ro] == 0 );
    }
  }
}