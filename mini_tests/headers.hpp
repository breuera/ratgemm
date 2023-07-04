#include <iostream>
#include <iomanip>
#include <cmath>

void print(const float* i_matrix, const float* o_matrix, int i_rows, int i_columns){
  std::cout << "Original Values\t Truncated Values\t Difference" << std::endl;
  std::cout << "---------------------------------------------------------------" << std::endl;
  float sum = 0;
  for (int i = 0; i < i_rows; i++) {
    for (int j = 0; j < i_columns; j++) {
      std::cout << std::fixed << std::setprecision(10);
      std::cout << i_matrix[i * i_columns + j] << "\t\t ";
      std::cout << o_matrix[i * i_columns + j]<< "\t\t ";
      std::cout << std::scientific << std::setprecision(2) << std::abs(i_matrix[i * i_columns + j] - o_matrix[i * i_columns + j]) << std::endl;
      std::cout << std::defaultfloat;
      sum += std::abs(i_matrix[i * i_columns + j] - o_matrix[i * i_columns + j]);
    }
  }
  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "Average difference: " << std::scientific << std::setprecision(2) << sum/(i_rows*i_columns) << std::endl;
}

float* generateRandomMatrix(int i_rows, int i_columns) {
  float* o_matrix = new float[i_rows * i_columns];
  srand(1);

  float l_minValue = -10.0;
  float l_maxValue = 10.0;

  for (int i = 0; i < i_rows; i++) {
    for (int j = 0; j < i_columns; j++) {
      float l_randomValue = l_minValue + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (l_maxValue - l_minValue)));
      o_matrix[i * i_columns + j] = l_randomValue;
    }
  }

  return o_matrix;
}

void printMatrix(const float* i_matrix, int i_rows, int i_columns) {
  for (int i = 0; i < i_rows; i++) {
    for (int j = 0; j < i_columns; j++) {
      std::cout << i_matrix[i * i_columns + j] << std::endl;;
    }
  }
}

void subtractMatrices(const float* in_1, const float* in_2, float* out, int i_rows, int i_columns) {
  for (int i = 0; i < i_rows; i++) {
    for (int j = 0; j < i_columns; j++) {
      out[i * i_columns + j] = in_1[i * i_columns + j] - in_2[i * i_columns + j];
    }
  }
}

void addMatrices(const float* i_matrix_A, const float* i_matrix_B, float* o_matrix, int i_rows, int i_columns) {
  for (int i = 0; i < i_rows; i++) {
    for (int j = 0; j < i_columns; j++) {
      *(o_matrix + i * i_columns + j) = *(i_matrix_A + i * i_columns + j) + *(i_matrix_B + i * i_columns + j);
    }
  }
}