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

void rat_gemm::backend::ConvertMatrix::float_to_bfloat16( float32_t i_value,
                                                          bfloat16_t* o_bf16_value ) {
  // Bit manipulation to extract sign, exponent, and mantissa of the input float
  float32_t l_float_bits = *reinterpret_cast<uint32_t*>(&i_value);
  float32_t l_sign_bit = l_float_bits >> 31;
  float32_t l_exponent_bits = (l_float_bits >> 23) & 0xFF;
  float32_t l_mantissa_bits = l_float_bits & 0x7FFFFF; // last 23 bits 0x7FFFFF

  // Calculate the bfloat16 representation
  // uint16_t l_bf16_value = (l_sign_bit << 15) | (l_exponent_bits - 127 + 15) << 10 | (l_mantissa_bits >> 16);
  bfloat16_t l_bf16_value = ((l_sign_bit << 15) | (l_exponent_bits << 7 ) | (l_mantissa_bits >> 16));
  // if (((l_sign_bit << 15) | (l_exponent_bits - 127 + 15) << 10 | (l_mantissa_bits >> 16)) == ((l_sign_bit << 15) | (l_exponent_bits << 7 ) | (l_mantissa_bits >> 16))){
  //     std::cout << "EQUAL" << std::endl;}
  // std::cout << (l_sign_bit) << std::endl;
  // std::cout << (l_exponent_bits) << std::endl;
  // std::cout << (l_mantissa_bits) << std::endl;

  // Store the bfloat16 value in the output parameter
  o_bf16_value = l_bf16_value;
}

void rat_gemm::backend::ConvertMatrix::float_to_two_bfloat16( float i_value,
                                                              bfloat16_t* o_bf16_value_1,
                                                              bfloat16_t* o_bf16_value_2 ){

  uint32_t l_float_bits = *reinterpret_cast<uint32_t*>(&i_value);
  uint32_t l_sign_bit = l_float_bits >> 31 & 0x1;
  uint16_t l_exponent_bits = (l_float_bits >> 23) & 0xFF;
  int32_t l_mantissa_bits = l_float_bits & 0x7FFFFF; // last 23 bits 0x7FFFFF

  // Apply rounding on mantissa if necessary
  l_mantissa_bits = (l_mantissa_bits + 0x7FFF + ((l_mantissa_bits >> 16) & 1)) >> 16;

  // Truncate the mantissa to 7 bits
  l_mantissa_bits &= 0x7F;

  // Adjust exponent bias
  int32_t l_adjusted_exponent = static_cast<int32_t>(l_exponent_bits) - 127 + 15;

  // Handle overflow and underflow
  if (l_adjusted_exponent < 0) {
      o_bf16_value_1 = static_cast<uint16_t>(l_sign_bit << 15);  // Set to positive or negative zero
      o_bf16_value_2 = 0;
  } else if (l_adjusted_exponent > 31) {
      o_bf16_value_1 = static_cast<uint16_t>((l_sign_bit << 15) | 0x7C00);  // Set to positive or negative infinity
      o_bf16_value_2 = 0;
  } else {
      o_bf16_value_1 = static_cast<uint16_t>((l_sign_bit << 15) | (l_adjusted_exponent << 10) | l_mantissa_bits);
      o_bf16_value_2 = static_cast<uint16_t>(l_float_bits >> 16);  // Shift the original value for the second BF16
  }
}


// static void convertToBFloat16(const float l_stiff_single[35][3*20], uint32_t l_stiff_bf16[35][3*20]) {
//   for (int64_t l_n = 0; l_n < 35; l_n++) {
//     for (int64_t l_m = 0; l_m < 3*20; l_m++) {
//       float value = l_stiff_single[l_n][l_m];
//       uint32_t bf16_value = float_to_bfloat16(value);
//       l_stiff_bf16[l_n][l_m] = bf16_value;
//     }
//   }
// }
