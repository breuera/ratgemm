BUILD_DIR ?= ./build
CXXFLAGS ?=
LDFLAGS ?=
RPATHS ?=
LIBXSMM_DIR ?= libxsmm
OPTIONS = -O2 -std=c++20 -pedantic -Wall -Wextra

CXXFLAGS += -I. -I${LIBXSMM_DIR}/include
LDFLAGS += ${LIBXSMM_DIR}/lib/libxsmm.a

$(info $$CXXFLAGS is [${CXXFLAGS}])
$(info $$LDFLAGS is [${LDFLAGS}])

all: test

bf16_matrix: src/backend/ConvertMatrix.cpp src/backend/ConvertMatrix.test.cpp
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/ConvertMatrix.cpp -o ${BUILD_DIR}/backend/ConvertMatrix.o
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/ConvertMatrix.test.cpp -o ${BUILD_DIR}/tests/backend/ConvertMatrix.test.o

rational_matrix: src/backend/RationalMatrix.cpp src/backend/RationalMatrix.test.cpp
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/RationalMatrix.cpp -o ${BUILD_DIR}/backend/RationalMatrix.o
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/RationalMatrix.test.cpp -o ${BUILD_DIR}/tests/backend/RationalMatrix.test.o

operations: src/backend/Operations.cpp src/backend/Operations.test.cpp
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/Operations.cpp -o ${BUILD_DIR}/backend/Operations.o
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/Operations.test.cpp -o ${BUILD_DIR}/tests/backend/Operations.test.o

rat_gemm: src/RatGemm.cpp src/RatGemm.test.cpp
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/RatGemm.cpp -o ${BUILD_DIR}/RatGemm.o
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/RatGemm.test.cpp -o ${BUILD_DIR}/tests/RatGemm.test.o

test: bf16_matrix rational_matrix operations rat_gemm
		$(CXX) ${CXXFLAGS} src/test.cpp ${BUILD_DIR}/*.o ${BUILD_DIR}/backend/*.o ${BUILD_DIR}/tests/*.test.o ${BUILD_DIR}/tests/backend/*.test.o ${LDFLAGS} -o ${BUILD_DIR}/test_all

$(shell mkdir -p build/backend)
$(shell mkdir -p build/tests/backend)