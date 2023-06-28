#include <iostream>
#include <iomanip>
#include <libxsmm.h>


int main() {
  const unsigned int length = 5;
  float input[length] = {1.23f, -0.45f, 2.56f, -3.67f,4.78f};
  libxsmm_bfloat16 output[length];

  libxsmm_truncate_convert_f32_bf16((const float*)input, (libxsmm_bfloat16*)output, length);

  // Print the original bfloat16 values and truncated bfloat16 values
  std::cout << "Original Values\t Truncated Values" << std::endl;
  std::cout << "--------------------------------" << std::endl;
  for (unsigned int i = 0; i < length; ++i) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << input[i] << "\t\t ";
    std::cout << output[i] << std::endl;
  }

  return 0;
}
