#include <arm_neon.h>

class BFloat16Converter {
public:
    BFloat16Converter() {}

    // Convert single-precision float to bf16 using ARM NEON intrinsics
    void float_to_bfloat16( float i_value,
                            uint32_t o_bf16_value ) {
        __asm__ __volatile__( // embed assembly code within C/C++ code without any reordering or optimization by the compiler
            "fcvtnt %w[out], %s[in]\n\t" // Convert float to bf16 using the fcvtnt instruction
            "sbfiz %w[out], %w[out], #16, #16\n\t" // Bitfield insert to truncate the result to 16 bits
            // Output constraint: "=r" means the output goes to a general-purpose register
            // Input constraint: "w" means the input comes from a general-purpose register
            : [out] "=r"(o_bf16_value) 
            : [in] "w"(i_value) 
        );
    }
};
