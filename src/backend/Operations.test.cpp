#include <catch2/catch.hpp>
#include "Operations.h"

TEST_CASE( "TODO: Write some description123",
           "[todo123]" ) {

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