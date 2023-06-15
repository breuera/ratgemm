#ifndef RAT_GEMM_BACKEND_CONVERT_MATRIX_H
#define RAT_GEMM_BACKEND_CONVERT_MATRIX_H

#include <arm_bf16.h>
#include <arm_neon.h>
#include <cstdint>

namespace rat_gemm {
  namespace backend {
    class ConvertMatrix;
  }
}

class rat_gemm::backend::ConvertMatrix {
public:
    ConvertMatrix() {}

    /**
     * Initializes an operation which gathers rows from A and stores them in B.
     *
     * @param i_value float value from stiff matrix A
     * @param o_bf16_value_1 first part of converted BF16 value from stiff matrix A
     * @param o_bf16_value_2 second part of the convereted Bf16 from stiff matrix A
     **/
    // Convert single-precision float to bf16
    void float_to_bfloat16( float32_t i_value,
                            bfloat16_t* o_bf16_value );

    void float_to_two_bfloat16( float value,
                                bfloat16_t* o_bf16_value_1,
                                bfloat16_t* o_bf16_value_2 );

    // // Conversion function for l_stiff_single to bf16 matrices
    // static void convertToBFloat16( const float l_stiff_single[35][3*20],
                                    //   uint32_t l_stiff_bf16[35][3*20] );
    
};

#endif