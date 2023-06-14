#include <iostream>
#include <catch2/catch.hpp>
#include "ConvertMatrix.h"


  // Example float value
  float value = 3.14159;
TEST_CASE( "test the convert version from float to BF16" ) {
  // Create an instance of the ConvertMatrix class
  rat_gemm::backend::ConvertMatrix conv;

  // Convert float to bf16
  uint16_t bf16_value_1;
  uint16_t bf16_value_2;
  conv.float_to_bfloat16( value,
                          bf16_value_1,
                          bf16_value_2 );

  // Display the converted bf16 value
  std::cout << "Original float value: " << value << std::endl;
  std::cout << "Converted bf16 value: " << bf16_value_1 << std::endl;
  REQUIRE( value == bf16_value_1 );
}