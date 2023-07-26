#include <catch2/catch.hpp>
#include <cstdlib>
#include <ctime>
#include "ConvertMatrix.h"

TEST_CASE("Test convert_fp32_two_bf16") {
    // Input values
    int i_s = 3;
    float* i_value = new float[i_s];
    float* l_diff = new float[i_s];

    // Call the function
    rat_gemm::backend::ConvertMatrix conv;
    conv.generateRandomMatrix(i_value, i_s);

    // Output arrays
    libxsmm_bfloat16* o_matrix_bf16_h1 = new libxsmm_bfloat16[i_s];
    libxsmm_bfloat16* o_matrix_bf16_h2 = new libxsmm_bfloat16[i_s];

    float* l_fp32_h1 = new float[i_s];
    float* l_fp32_h2 = new float[i_s];


    conv.convert_fp32_two_bf16(i_value, o_matrix_bf16_h1, o_matrix_bf16_h2, i_s);

    libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)o_matrix_bf16_h1, (float*)l_fp32_h1, i_s);
    libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)o_matrix_bf16_h2, (float*)l_fp32_h2, i_s);

    conv.diff(i_value, l_fp32_h1, l_fp32_h2, l_diff, i_s);

    float l_expectedValue = 0;
    float l_t = 1e-4; 

    // Verify the results
    for (int i = 0;  i < i_s ; ++i){
      REQUIRE(std::abs((l_expectedValue) - (l_diff[i])) <= (l_t));
    }

    // Clean up
    delete[] i_value;
    delete[] o_matrix_bf16_h1;
    delete[] o_matrix_bf16_h2;
    delete[] l_diff;
    delete[] l_fp32_h1;
    delete[] l_fp32_h2;
}
