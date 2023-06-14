#include "ConvertMatrix.h"
#include <iostream>

// Convert single-precision float to bf16 using ARM NEON intrinsics
// void rat_gemm::backend::ConvertMatrix::float_to_bfloat16( float i_value,
//                                                       uint32_t o_bf16_value ) {
//   __asm__ __volatile__( // embed assembly code within C/C++ code without any reordering or optimization by the compiler
//     "fcvtnt %w[out], %s[in]\n\t" // Convert float to bf16 using the fcvtnt instruction
//     "sbfiz %w[out], %w[out], #16, #16\n\t" // Bitfield insert to truncate the result to 16 bits
//     // Output constraint: "=r" means the output goes to a general-purpose register
//     // Input constraint: "w" means the input comes from a general-purpose register
//     : [out] "=r"(o_bf16_value) 
//     : [in] "w"(i_value) 
//   );
// }

void rat_gemm::backend::ConvertMatrix::float_to_bfloat16( float i_value,
                                                          uint16_t& o_bf16_value_1,
                                                          uint16_t& o_bf16_value_2 ) {
  // Bit manipulation to extract sign, exponent, and mantissa of the input float
  uint32_t l_float_bits = *reinterpret_cast<uint32_t*>(&i_value);
  uint16_t l_sign_bit = l_float_bits >> 31;
  uint16_t l_exponent_bits = (l_float_bits >> 23) & 0xFF;
  uint32_t l_mantissa_bits = l_float_bits & 0x7FFFFF; // last 23 bits 0x7FFFFF

  // Calculate the bfloat16 representation
  // uint16_t l_bfloat16_value_1 = (l_sign_bit << 15) | (l_exponent_bits - 127 + 15) << 10 | (l_mantissa_bits >> 16);
  // std::cout << (l_sign_bit) << std::endl;
  // std::cout << (l_exponent_bits) << std::endl;
  // std::cout << (l_mantissa_bits) << std::endl;
  std::cout << ((l_sign_bit << 15) | (l_exponent_bits << 7 ) | (l_mantissa_bits >> 16)) << std::endl;
  


  // Store the bfloat16 value in the output parameter
  // o_bf16_value_1 = l_bfloat16_value_1;
}
