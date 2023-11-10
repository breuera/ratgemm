#include <iostream>
#include <iostream>

int main() {
    unsigned int uintValue = 0x4dfb0297; // Replace 0x40490FDB with your uint value

    // Create a pointer to the unsigned int value
    unsigned int* uintPointer = &uintValue;

    // Create a float pointer and reinterpret the bits from the uint pointer
    float* floatPointer = reinterpret_cast<float*>(uintPointer);

    // Access the float value without changing the bits
    float floatValue = *floatPointer;

    // Print the original uint and the interpreted float
    std::cout << "Unsigned int value (uint): " << uintValue << std::endl;
    std::cout << "Float value (reinterpret_cast): " << floatValue << std::endl;

    return 0;
}
