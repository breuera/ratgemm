#include <iostream>
#include <iomanip>
#include <libxsmm.h>

int main() {
  const unsigned int length = 5;

  libxsmm_datatype fp32 = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype bf16 = LIBXSMM_DATATYPE_BF16;

  float input[length] = {1.23f, -0.45f, 2.56f, -3.67f, 4.78f};
  void* first_part_bf16;
  void* first_part_fp32 = malloc(length * LIBXSMM_TYPESIZE(fp32));

  libxsmm_truncate_convert_f32_bf16((const float*)input, (libxsmm_bfloat16*)first_part_bf16, length);

  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)first_part_bf16, (float*)first_part_fp32, length);

  // Print the original bfloat16 values and truncated bfloat16 values
  std::cout << "Original Values\t Truncated Values" << std::endl;
  std::cout << "--------------------------------" << std::endl;
  for (unsigned int i = 0; i < length; ++i) {
    std::cout << std::fixed << std::setprecision(10);
    std::cout << input[i] << "\t\t ";
    std::cout << ((float*)first_part_fp32[i]) << std::endl;
  }

  free(first_part_fp32);

  return 0;
}
