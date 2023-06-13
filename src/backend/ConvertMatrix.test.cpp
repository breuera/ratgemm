#include <iostream>
#include <iomanip>
#include <catch2/catch.hpp>
#include "ConvertMatrix.h"


int main() {
    // Example float value
    float value = 3.14159;
    
    // Create an instance of the BFloat16Converter class
    rat_gemm::backend::Converter cov;

    // Convert float to bf16
    uint32_t bf16_value = cov.float_to_bfloat16(value);

    // Display the converted bf16 value
    //float converted_value = cov.bfloat16_to_float(bf16_value);
    std::cout << "Original float value: " << value << std::endl;
    std::cout << "Converted bf16 value: " << bf16_value << std::endl;
    REQUIRE( value == bf16_value );
    
    return 0;
}