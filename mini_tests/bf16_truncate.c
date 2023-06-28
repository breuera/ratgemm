#include <iostream>
#include <iomanip>
#include <libxsmm.h>
#include <cmath>


void print(const float* input, const float* output, unsigned int length){
  std::cout << "Original Values\t Truncated Values" << std::endl;
  std::cout << "--------------------------------" << std::endl;
  for (unsigned int i = 0; i < length; ++i) {
    // output[i] = input[i];
    std::cout << std::fixed << std::setprecision(10);
    std::cout << input[i] << "\t\t ";
    std::cout << output[i] << std::endl;
  }
}

int main() {
  const unsigned int length = 5;
  float input[length] = {1.23f, -0.45f, 2.56f, -3.67f,4.78f};
  float second_half_input[length];

  libxsmm_bfloat16 first_half_bf16[length];
  libxsmm_bfloat16 second_half_bf16[length];

  float first_half_fp32[length];
  float second_half_fp32[length];
  
  libxsmm_truncate_convert_f32_bf16((const float*)input, (libxsmm_bfloat16*)first_half_bf16, length);
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)first_half_bf16, first_half_fp32, length);

  // Print the original bfloat16 values and truncated bfloat16 values
  print(input, first_half_fp32, length);
  
  for (unsigned int i = 0; i < length; ++i) {
    second_half_input[i] = input[i] - first_half_fp32[i];
  }
  
  libxsmm_rne_convert_fp32_bf16((const float*)second_half_input, (libxsmm_bfloat16*)second_half_bf16, length);
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)second_half_bf16, second_half_fp32, length);

  print(second_half_input, second_half_fp32, length);

  for (unsigned int i = 0; i < length; ++i) {
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Main fp32:             " << std::defaultfloat << std::setprecision(10) << input[i] << std::endl;
    std::cout << "Converted fp32 2steps: " << std::defaultfloat << std::setprecision(10) << first_half_fp32[i] + second_half_fp32[i] << std::endl;
    std::cout << "Loss:                  " << std::scientific << std::setprecision(2) << std::fabs(input[i] - (first_half_fp32[i] + second_half_fp32[i])) << std::endl;
    std::cout << std::defaultfloat;

    // libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)(first_half_bf16 + second_half_bf16), second_half_input, length);
    // std::cout << "converted fp32 once now" << second_half_input[i] << std::endl;
  }


  return 0;
}
