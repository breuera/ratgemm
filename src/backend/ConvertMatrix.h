#ifndef RAT_GEMM_BACKEND_CONVERT_MATRIX_H
#define RAT_GEMM_BACKEND_CONVERT_MATRIX_H

// #include <arm_neon.h>
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
    void float_to_bfloat16( float i_value,
                            uint16_t& o_bf16_value_1,
                            uint16_t& o_bf16_value_2 );

/*
    // Convert bf16 to single-precision float using ARM NEON intrinsics
    static inline float bfloat16_to_float(uint32_t bf16_value) {
        float value;
        __asm__ __volatile__(
            "mov %w[out], %w[in]\n\t"
            "sbfm %w[out], %w[out], #16, #16\n\t"
            "fmov %s[out], %w[out]\n\t"
            : [out] "=w"(value)
            : [in] "r"(bf16_value)
        );
        return value;
    }

    // Conversion function for l_stiff_single to bf16 matrices
    static void convertToBFloat16(const float l_stiff_single[35][3*20], uint32_t l_stiff_bf16[35][3*20]) {
        for (int64_t l_n = 0; l_n < 35; l_n++) {
            for (int64_t l_m = 0; l_m < 3*20; l_m++) {
                float value = l_stiff_single[l_n][l_m];
                uint32_t bf16_value = float_to_bfloat16(value);
                l_stiff_bf16[l_n][l_m] = bf16_value;
            }
        }
    }
    */
};

#endif