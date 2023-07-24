#include <iostream>
#include <libxsmm.h>

void test_truncate_convert_f32_bf16(const float* in, libxsmm_bfloat16* out, unsigned int length) {
  unsigned int i = 0;

  /* truncate buffer to bf16 */
  for ( i = 0; i < length; ++i ) {
    libxsmm_float_uint hybrid_in = { 0 };
    libxsmm_bfloat16 res;

    hybrid_in.f = in[i];

    /* DAZ */
    hybrid_in.u = ( (hybrid_in.u & 0x7f800000) == 0x0 ) ? ( hybrid_in.u & 0x80000000 ) : hybrid_in.u;
    /* we do not round inf and NaN */
    hybrid_in.u = ( (hybrid_in.u & 0x7f800000) == 0x7f800000 ) ? ( ((hybrid_in.u & 0x007fffff) == 0x0) ? hybrid_in.u : hybrid_in.u | 0x00400000 ) : hybrid_in.u;
    /* shift right */
    res = (unsigned short)(hybrid_in.u >> 16);

    out[i] = res;
  }
}

void test_convert_bf16_f32(const libxsmm_bfloat16* in, float* out, unsigned int length) {
  unsigned int i = 0;

  /* up-convert is super simple */
  for ( i = 0; i < length; ++i ) {
    libxsmm_float_uint hybrid_in = { 0 };

    hybrid_in.u = in[i];
    /* DAZ */
    hybrid_in.u = ( (hybrid_in.u & 0x7f80) == 0x0 ) ? (unsigned short)(hybrid_in.u & 0x8000) : hybrid_in.u;
    hybrid_in.u = hybrid_in.u << 16;

    out[i] = hybrid_in.f;
  }
}

int main() {
  unsigned int length = 4;
  float input[] = { 1.7845f, -2.1458f, 0.6254f, 0.7941f };
  libxsmm_bfloat16 input_bf16[length];
  float output[length]; 

  float diff[length];

  libxsmm_bfloat16 diff_bf16[length];
  float output_diff[length];

  test_truncate_convert_f32_bf16(input, input_bf16, length);
  test_convert_bf16_f32(input_bf16, output, length);

  std::cout << "first:" << std::endl;

  for (unsigned int i = 0; i < length; ++i) {
    std::cout << output[i] << std::endl;
    diff[i] = input[i] - output[i]; 
  }

  test_truncate_convert_f32_bf16(diff, diff_bf16, length);
  test_convert_bf16_f32(diff_bf16, output_diff, length);

  std::cout << "\ndiff:" << std::endl;

  for (unsigned int i = 0; i < length; ++i) {
    std::cout << output_diff[i] << std::endl;
  }

  return 0;
}