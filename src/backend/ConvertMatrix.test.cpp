#include <catch2/catch.hpp>
#include <cstdlib>
#include <ctime>
#include "ConvertMatrix.h"

TEST_CASE("Test float_to_two_bfloat16") {
    // Input values
    int i_s = 3;
    float* i_value = new float[i_s];

    // Seed the random number generator
    std::srand(std::time(0));

    // Populate the input array with random numbers
    for (int i = 0; i < i_s; ++i) {
      i_value[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // Output arrays
    libxsmm_bfloat16* o_bf16_value_1 = new libxsmm_bfloat16[i_s];
    libxsmm_bfloat16* o_bf16_value_2 = new libxsmm_bfloat16[i_s];

    // Call the function
    rat_gemm::backend::ConvertMatrix conv;
    conv.float_to_two_bfloat16(i_value, i_s, o_bf16_value_1, o_bf16_value_2);

    // Verify the results
    for (int i = 0; i < i_s; ++i) {
      float tmp = static_cast<float>(o_bf16_value_1[i]);
      float diff = i_value[i] - tmp;
      float tmp2 = static_cast<float>(o_bf16_value_2[i]);

      // Use the Approx matcher for "almost equal" comparison
      REQUIRE(diff == Approx(tmp2));
    }

    // Clean up
    delete[] i_value;
    delete[] o_bf16_value_1;
    delete[] o_bf16_value_2;
}
