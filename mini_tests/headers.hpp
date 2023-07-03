
#include <iostream>
#include <iomanip>

void print(const float* input, const float* output, int rows, int columns){
  std::cout << "Original Values\t Truncated Values\t Difference" << std::endl;
  std::cout << "---------------------------------------------------------------" << std::endl;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      std::cout << std::fixed << std::setprecision(10);
      std::cout << input[i * columns + j] << "\t\t ";
      std::cout << output[i * columns + j]<< "\t\t ";
      std::cout << std::scientific << std::setprecision(2) << input[i * columns + j] - output[i * columns + j] << std::endl;
      std::cout << std::defaultfloat;
    }
  }
}

float** generateRandomMatrix(int rows, int columns) {
  float** matrix = new float*[rows];
  srand(static_cast<unsigned int>(time(0)));

  float minValue = -10.0;
  float maxValue = 10.0;

  for (int i = 0; i < rows; i++) {
    matrix[i] = new float[columns];
    for (int j = 0; j < columns; j++) {
      float randomValue = minValue + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (maxValue - minValue)));
      std::cout << randomValue << std::endl;
      matrix[i][j] = randomValue;
    }
  }
  
  return matrix;
}

void freeMatrix(float** matrix, int rows) {
  for (int i = 0; i < rows; i++) {
    delete[] matrix[i];
  }
  delete[] matrix;
}

void printMatrix(const float* matrix, int rows, int columns) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      std::cout << matrix[i * columns + j] << std::endl;;
    }
  }
}

void subtractMatrices(const float* in_1, const float* in_2, float* out, int rows, int columns) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      out[i * columns + j] = in_1[i * columns + j] - in_2[i * columns + j];
    }
  }
}

void addMatrices(const float* matrixA, const float* matrixB, float* result, int rows, int columns) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      *(result + i * columns + j) = *(matrixA + i * columns + j) + *(matrixB + i * columns + j);
    }
  }
}