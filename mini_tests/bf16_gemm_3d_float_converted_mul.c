#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <iomanip> // For setting precision when printing floating-point values
#include <random> // For random number generation
#include <libxsmm.h>

double stiff_test[3][35][35] = {
  {
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  2.5,-94.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  2.5,10.5,-52.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,31.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.5,-73.5,0.0,0.0,84.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.5,-38.5,-17.5,0.0,-16.0,50.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.5,10.5,-42.0,0.0,1.5,-6.0,10.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,10.5,0.0,0.0,0.0,-30.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,10.5,0.0,0.0,0.0,1.5,-21.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.5,10.5,10.5,0.0,1.5,1.5,0.5,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.0,-54.0,0.0,0.0,81.0,0.0,0.0,0.0,0.0,0.0,-75.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.0,-39.0,-7.5,0.0,24.75,28.125,0.0,0.0,0.0,0.0,18.75,-46.875,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.0,-18.0,-18.0,0.0,-14.625,36.0,2.625,0.0,0.0,0.0,-3.125,10.0,-91.875,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.0,9.0,-31.5,0.0,2.25,-7.875,10.5,0.0,0.0,0.0,0.25,-0.875,10.5,-21.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,4.5,0.0,0.0,0.0,-16.875,0.0,0.0,0.0,0.0,0.0,0.0,84.375,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,4.5,0.0,0.0,0.0,-9.0,-5.25,0.0,0.0,0.0,0.0,0.0,-7.5,61.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,4.5,0.0,0.0,0.0,1.125,-12.0,0.0,0.0,0.0,0.0,0.0,0.375,-4.0,9.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.0,-18.0,4.5,0.0,-14.625,-9.0,0.125,0.0,0.0,1.25,-3.125,-2.5,-4.375,0.0,0.0,0.0,0.0,-43.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.0,9.0,-9.0,0.0,2.25,-2.25,-3.25,0.0,0.0,1.25,0.25,-0.25,-3.25,-2.25,0.0,0.0,0.0,1.25,-11.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,15.0,0.0,0.0,0.0,3.75,3.75,0.0,0.0,0.0,0.0,0.0,1.25,1.25,0.25,0.0,0.0,1.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
  },
  {
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  7.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  2.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  3.75,-141.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  -2.5,-10.5,-105.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.25,-47.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.25,5.25,-26.25,26.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.25,5.25,5.25,15.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  2.25,-110.25,0.0,0.0,126.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  -1.5,38.5,-35.0,0.0,16.0,100.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  3.75,26.25,26.25,0.0,3.75,3.75,26.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,-36.75,0.0,0.0,42.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,-19.25,-8.75,8.75,-8.0,25.0,0.0,-25.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,5.25,-21.0,-15.75,0.75,-3.0,5.25,-2.25,-21.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,-19.25,1.75,5.25,-8.0,-5.0,0.0,-15.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,5.25,-10.5,5.25,0.75,-1.5,-1.75,0.75,-10.5,3.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,5.25,5.25,10.5,0.75,0.75,0.25,1.5,1.5,2.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.5,-81.0,0.0,0.0,121.5,0.0,0.0,0.0,0.0,0.0,-112.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  -1.0,39.0,-15.0,0.0,-24.75,56.25,0.0,0.0,0.0,0.0,-18.75,-93.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  2.5,-45.0,11.25,0.0,-36.5625,-22.5,6.5625,0.0,0.0,0.0,-7.8125,-6.25,-229.6875,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  -2.0,-18.0,-63.0,0.0,-4.5,-15.75,-5.25,0.0,0.0,0.0,-0.5,-1.75,-5.25,-63.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-27.0,0.0,0.0,40.5,0.0,0.0,0.0,0.0,0.0,-37.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-19.5,-3.75,3.75,12.375,14.0625,0.0,-14.0625,0.0,0.0,9.375,-23.4375,0.0,0.0,70.3125,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-9.0,-9.0,-6.75,-7.3125,18.0,1.3125,13.5,-5.25,0.0,-1.5625,5.0,-45.9375,0.0,11.25,61.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,4.5,-15.75,15.75,1.125,-3.9375,5.25,3.9375,10.5,0.0,0.125,-0.4375,5.25,-10.5,1.3125,3.5,10.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-19.5,0.75,2.25,12.375,-2.8125,0.0,-8.4375,0.0,0.0,9.375,4.6875,0.0,0.0,42.1875,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-9.0,-4.5,2.25,-7.3125,9.0,-0.4375,-4.5,-2.625,0.875,-1.5625,2.5,15.3125,0.0,-3.75,30.625,0.0,-30.625,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,4.5,-11.25,2.25,1.125,-2.8125,1.25,0.5625,-6.0,-2.5,0.125,-0.3125,1.25,4.5,0.1875,-2.0,4.5,-2.5,-9.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-9.0,2.25,4.5,-7.3125,-4.5,0.0625,-9.0,0.375,0.625,-1.5625,-1.25,-2.1875,0.0,-7.5,-4.375,0.0,-21.875,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,4.5,-4.5,4.5,1.125,-1.125,-1.625,1.125,-6.375,0.625,0.125,-0.125,-1.625,-1.125,0.375,-2.125,-1.125,0.625,-5.625,1.125,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,4.5,4.5,7.5,1.125,1.125,0.375,1.875,1.875,1.875,0.125,0.125,0.375,0.125,0.625,0.625,0.125,1.875,0.625,0.875,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
  },
  {
    {  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  2.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  2.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  -6.25,-236.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.25,-47.25,-94.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.25,5.25,-42.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.25,-47.25,0.0,31.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.25,5.25,-26.25,5.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  1.25,5.25,5.25,15.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  8.25,110.25,0.0,0.0,252.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,-36.75,73.5,0.0,42.0,105.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,-19.25,-14.0,0.0,-8.0,40.0,20.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,5.25,-26.25,0.0,0.75,-3.75,11.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,-36.75,0.0,-24.5,42.0,0.0,0.0,-35.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,-19.25,-8.75,1.75,-8.0,25.0,0.0,-5.0,-20.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,5.25,-21.0,1.75,0.75,-3.0,5.25,0.25,-6.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,-19.25,1.75,5.25,-8.0,-5.0,0.0,-15.0,0.0,4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,5.25,-10.5,5.25,0.75,-1.5,-1.75,0.75,-10.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.75,5.25,5.25,10.5,0.75,0.75,0.25,1.5,1.5,2.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  -7.0,-189.0,0.0,0.0,-94.5,0.0,0.0,0.0,0.0,0.0,-262.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-27.0,-81.0,0.0,40.5,-67.5,0.0,0.0,0.0,0.0,-37.5,-112.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-19.5,-6.0,0.0,12.375,22.5,-16.875,0.0,0.0,0.0,9.375,-37.5,-196.875,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-9.0,-11.25,0.0,-7.3125,22.5,2.8125,0.0,0.0,0.0,-1.5625,6.25,-98.4375,-43.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,4.5,-18.0,0.0,1.125,-4.5,7.875,0.0,0.0,0.0,0.125,-0.5,7.875,-28.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-27.0,0.0,27.0,40.5,0.0,0.0,22.5,0.0,0.0,-37.5,0.0,0.0,0.0,112.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-19.5,-3.75,0.75,12.375,14.0625,0.0,-2.8125,16.875,0.0,9.375,-23.4375,0.0,0.0,14.0625,65.625,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-9.0,-9.0,0.75,-7.3125,18.0,1.3125,-1.5,-1.5,0.0,-1.5625,5.0,-45.9375,0.0,-1.25,17.5,8.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,4.5,-15.75,0.75,1.125,-3.9375,5.25,0.1875,-2.625,0.0,0.125,-0.4375,5.25,-10.5,0.0625,-0.875,3.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-19.5,0.75,2.25,12.375,-2.8125,0.0,-8.4375,0.0,-3.375,9.375,4.6875,0.0,0.0,42.1875,0.0,0.0,-39.375,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-9.0,-4.5,2.25,-7.3125,9.0,-0.4375,-4.5,-2.625,0.125,-1.5625,2.5,15.3125,0.0,-3.75,30.625,0.0,-4.375,-8.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,4.5,-11.25,2.25,1.125,-2.8125,1.25,0.5625,-6.0,0.125,0.125,-0.3125,1.25,4.5,0.1875,-2.0,4.5,0.125,-2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,-9.0,2.25,4.5,-7.3125,-4.5,0.0625,-9.0,0.375,0.625,-1.5625,-1.25,-2.1875,0.0,-7.5,-4.375,0.0,-21.875,0.0,1.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,4.5,-4.5,4.5,1.125,-1.125,-1.625,1.125,-6.375,0.625,0.125,-0.125,-1.625,-1.125,0.375,-2.125,-1.125,0.625,-5.625,0.125,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
    {  0.5,4.5,4.5,7.5,1.125,1.125,0.375,1.875,1.875,1.875,0.125,0.125,0.375,0.125,0.625,0.625,0.125,1.875,0.625,0.875,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  },
  },
};

float bfloat16_to_float(libxsmm_bfloat16 bf16_value) {
    uint16_t bf16_bits = static_cast<uint16_t>(bf16_value);
    uint32_t fp32_bits = ((bf16_bits & 0x7FFF) << 16) | ((bf16_bits & 0x8000) << 16) | ((bf16_bits & 0x7F) << 13);
    float float_value;
    memcpy(&float_value, &fp32_bits, sizeof(float));
    return float_value;
}

void bf16_gemm(libxsmm_bfloat16* i_dof_1,
               libxsmm_bfloat16* i_dof_2,
               libxsmm_bfloat16* i_stiff,
               float* o_result,
               libxsmm_blasint i_m, //9
               libxsmm_blasint i_n, // 3*20
               libxsmm_blasint i_k){ //35
  for (int row = 0; row < i_m; row++) {
    for (int col = 0; col < i_n; col++) {
      float acc = 0.f;
      for (int k = 0; k < i_k; k++) {
        float val_dof_1_fp32 = bfloat16_to_float(i_dof_1[row * i_k + k]);
        float val_dof_2_fp32 = bfloat16_to_float(i_dof_2[row * i_k + k]);
        float val_stiff = bfloat16_to_float(i_stiff[k * i_n + col]);
        acc += (val_dof_1_fp32 * val_stiff)+ (val_dof_2_fp32 * val_stiff);
      }
      o_result[row * i_n + col] += acc;
    }
  }
}

void bf16_gemm_one(libxsmm_bfloat16* i_dof,
                   libxsmm_bfloat16* i_stiff,
                   float* o_result,
                   libxsmm_blasint i_di, //3
                   libxsmm_blasint i_m, //9
                   libxsmm_blasint i_n, //20
                   libxsmm_blasint i_k, //35
                   libxsmm_blasint i_nz_size){ //10 in this case
   for( int64_t l_di = 0; l_di < i_di; l_di++ ) {
    for( int64_t l_m = 0; l_m < i_n; l_m++ ) {
      for( int64_t l_n = 0; l_n < i_m; l_n++ ) {
        float acc = 0.0f;
        for( int64_t l_k = 0; l_k < (i_k + i_nz_size); l_k++ ) {
          acc += bfloat16_to_float(i_dof[l_n * (i_k + i_nz_size) + l_k]) * bfloat16_to_float(i_stiff[l_di * i_n + l_k * i_n * i_di + l_m]);
          // std::cout << l_n * (i_k + i_nz_size) + l_k << " * " << l_di * i_n + l_k * i_n * i_di + l_m << " = " <<  bfloat16_to_float(i_dof[l_n * (i_k + i_nz_size) + l_k]) << " * " << bfloat16_to_float(i_stiff[l_di * i_n + l_k * i_n * i_di + l_m]) << std::endl; 
        }
        o_result[l_di * i_m * i_n + l_n * i_n + l_m] += acc;
        // std::cout << " = " << l_di * i_m * i_n + l_n * i_n + l_m; 
        // std::cout << " = " << acc << std::endl; 
      }
    }
  }
}

float upconvert_bf16(libxsmm_bfloat16 x) {
  libxsmm_bfloat16_f32 bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

void bf16_gemm_upconvert(libxsmm_bfloat16* i_dof,
                         libxsmm_bfloat16* i_stiff,
                         float* o_result,
                         libxsmm_blasint i_m, // 9
                         libxsmm_blasint i_n, // 3 * 20
                         libxsmm_blasint i_k) { // 35
  libxsmm_blasint i, j, l;
  for (j = 0; j < i_n; j++) {
    for (i = 0; i < i_m; i++) {
      float acc = 0.f;
      for (l = 0; l < i_k; l++) {
        float val_dof = upconvert_bf16(i_dof[i + l * i_m]); // Convert i_dof to float
        float val_stiff = upconvert_bf16(i_stiff[l + j * i_k]); // Convert i_stiff to float
        acc += val_dof * val_stiff;
      }
      o_result[i + j * i_m] += acc;
    }
  }
}

void printMatrix(const libxsmm_bfloat16* matrix,
                 int rows,
                 int cols) {

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      // Indexing formula for a 2D matrix: element at row i, column j is matrix[i * cols + j]
      std::cout << std::fixed << std::setprecision(2) << float(matrix[i * cols + j]) << " ";
    }
    std::cout << std::endl;
  }
}

void convert_fp32_two_bf16(const float* i_matrix,
                           libxsmm_bfloat16* o_matrix_bf16_h1,
                           libxsmm_bfloat16* o_matrix_bf16_h2,
                           int i_s){

  float* l_first_half_fp32 = new float[i_s];
  float* l_second_half_input = new float[i_s];

  libxsmm_truncate_convert_f32_bf16((const float*)i_matrix, (libxsmm_bfloat16*)o_matrix_bf16_h1, i_s);
  libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)o_matrix_bf16_h1, (float*)l_first_half_fp32, i_s);

  for (int i = 0; i < i_s; i++) {
      l_second_half_input[i] = i_matrix[i] - l_first_half_fp32[i];
  }

  libxsmm_truncate_convert_f32_bf16((const float*)l_second_half_input, (libxsmm_bfloat16*)o_matrix_bf16_h2, i_s);

  delete[] l_first_half_fp32;
  delete[] l_second_half_input;
}

void RowNonZero(const std::vector<libxsmm_bfloat16>& i_vec,
                int i_m,
                int i_n,
                std::vector<int64_t>& o_nz_idx) {

  o_nz_idx.clear();

  for (int i = 0; i < i_m; ++i) {
    bool l_rowHasNonZero = false;
    for (int j = 0; j < i_n; ++j) {
      if (i_vec[i * i_n + j] != 0) {
          l_rowHasNonZero = true;
          break;
      }
    }
    if (l_rowHasNonZero) {
      o_nz_idx.push_back(i);
    }
  }
}

void printAsMatrix(const std::vector<libxsmm_bfloat16>& vec,
                   int i_m,
                   int i_n) {

  if (vec.size() != i_m * i_n) {
    std::cout << "Error: The size of the vector does not match the specified matrix dimensions.\n";
    return;
  }

  for (int i = 0; i < i_m; ++i) {
    for (int j = 0; j < i_n; ++j) {
      std::cout << bfloat16_to_float(vec[i * i_n + j]) << " ";
    }
    std::cout << std::endl;
  }
}

void pad_rows(const std::vector<libxsmm_bfloat16>& i_vec_1,
              const std::vector<libxsmm_bfloat16>& i_vec_2,
              const std::vector<int64_t>& i_indices,
              std::vector<libxsmm_bfloat16>& o_mat_padded,
              const int64_t i_m,
              const int64_t i_n) {

  // Copy main values
  for (int64_t l_n = 0; l_n < i_n; l_n++) {
    for (int64_t l_m = 0; l_m < i_m; l_m++) {
      o_mat_padded.push_back(i_vec_1[l_m + i_m * l_n]);
    }
  }
  // Append non-zero row
  for (std::size_t l_co_in = 0; l_co_in < i_indices.size(); l_co_in++) {
    for (int64_t l_n = 0; l_n < i_n; l_n++) {
      int64_t l_id_in = i_n * i_indices[l_co_in] + l_n;
      o_mat_padded.push_back(i_vec_2[l_id_in]);
    }
  }
}

void pad_cols(const std::vector<libxsmm_bfloat16>& io_vec_1,
              const std::vector<libxsmm_bfloat16>& i_vec_2,
              const std::vector<int64_t>& i_indices,
              std::vector<libxsmm_bfloat16>& o_mat_padded,
              const int64_t i_m,
              const int64_t i_n) {

  // Copy main values
  for (int64_t l_m = 0; l_m < i_m; l_m++) {
    for (int64_t l_n = 0; l_n < i_n; l_n++) {
      o_mat_padded.push_back(io_vec_1[l_m * i_n + l_n]);
    }
    // Append non-zero row
    for (std::size_t l_co_in = 0; l_co_in < i_indices.size(); l_co_in++) {
      int64_t l_id_in = i_indices[l_co_in] + (l_m * i_n);
      o_mat_padded.push_back(i_vec_2[l_id_in]);
    }
  }
}

int main() {

  // Select the elements to keep 3 * 35 * 20
  std::vector<std::vector<std::vector<double>>> new_stiff_test(3, std::vector<std::vector<double>>(35, std::vector<double>(20)));

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 35; j++) {
      for (int k = 0; k < 20; k++) {
        new_stiff_test[i][j][k] = stiff_test[i][j][k];
      }
    }
  }

  float l_stiff_fp32[3][35][20];

  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    for( int64_t l_n = 0; l_n < 35; l_n++ ) {
      for( int64_t l_m = 0; l_m < 20; l_m++ ) {
        l_stiff_fp32[l_di][l_n][l_m] = new_stiff_test[l_di][l_n][l_m];
      }
    }
  }

  // adjust data layout to single matrix with 3*M
  float l_stiff_single[35][3*20];
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    for( int64_t l_n = 0; l_n < 35; l_n++ ) {
      for( int64_t l_m = 0; l_m < 20; l_m++ ) {
        int64_t l_m_single = l_di*20 + l_m;
        l_stiff_single[l_n][l_m_single] = l_stiff_fp32[l_di][l_n][l_m];
      }
    }
  }

  float l_dofs[9][35] = { 0 };

  for( int64_t l_qt = 0; l_qt < 9; l_qt++ ) {
    for( int64_t l_md = 0; l_md < 35; l_md++ ) {
      l_dofs[l_qt][l_md] = (float) (rand()) / (float) (RAND_MAX);
    }
  }
  // // Display the main stiff with depth 3
  // for (int i = 0; i < 3; i++) {
  //   for (int j = 0; j < 35; j++) {
  //     for (int k = 0; k < 20; k++) {
  //       std::cout << new_stiff_test[i][j][k] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }

  // // Display the converted to 2d
  // for (int i = 0; i < 35; i++) {
  //   for (int j = 0; j < 3*20; j++) {
  //     std::cout << l_stiff_single[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // convert stiff and dof to two bf16
  libxsmm_bfloat16* l_dof_1 = new libxsmm_bfloat16[9 * 35];
  libxsmm_bfloat16* l_dof_2 = new libxsmm_bfloat16[9 * 35];

  libxsmm_bfloat16* l_stiff_1 = new libxsmm_bfloat16[35 * 3*20];
  libxsmm_bfloat16* l_stiff_2 = new libxsmm_bfloat16[35 * 3*20];

  convert_fp32_two_bf16((const float *)l_stiff_single, l_stiff_1, l_stiff_2, 35 * 3*20);
  convert_fp32_two_bf16((const float *)l_dofs, l_dof_1, l_dof_2, 9 * 35);

  // // Display the values in float converted to bf16
  // std::cout << "\n-----------------------Converted to bf16:--------------------------------\n";
  // for (int i = 0; i < 35; i++) {
  //   for (int j = 0; j < 3*20; j++) {
  //     std::cout << bfloat16_to_float(l_stiff_1[i * (3*20) + j]) << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // std::cout << "\n-----------------------Converted to bf16 part two with many zeros:--------------------------------\n";
  // for (int i = 0; i < 35; i++) {
  //   for (int j = 0; j < 3*20; j++) {
  //     std::cout << bfloat16_to_float(l_stiff_2[i * (3*20) + j]) << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // get non zero rows and do padding 
  std::vector<libxsmm_bfloat16> l_stiff_1_vec(l_stiff_1, l_stiff_2 + 35 * 3*20);
  std::vector<libxsmm_bfloat16> l_stiff_2_vec(l_stiff_2, l_stiff_2 + 35 * 3*20);

  std::vector<int64_t> l_nz_idx;
  RowNonZero(l_stiff_2_vec, 35, 3*20, l_nz_idx);

  std::vector<libxsmm_bfloat16> l_stiff_padded;
  pad_rows(l_stiff_1_vec, l_stiff_2_vec, l_nz_idx, l_stiff_padded, 35, 3*20);
  
  // std::cout << "\n-----------------------Padded stiff:--------------------------------\n";
  // printAsMatrix(l_stiff_padded, 35 + l_nz_idx.size(), 3*20);

  std::vector<libxsmm_bfloat16> l_dof_1_vec(l_dof_1, l_dof_1 + 9 * 35);
  std::vector<libxsmm_bfloat16> l_dof_2_vec(l_dof_2, l_dof_2 + 9 * 35);

  std::vector<libxsmm_bfloat16> l_dof_1_padded;
  std::vector<libxsmm_bfloat16> l_dof_2_padded;

  pad_cols(l_dof_1_vec, l_dof_1_vec, l_nz_idx, l_dof_1_padded, 9, 35);
  pad_cols(l_dof_2_vec, l_dof_2_vec, l_nz_idx, l_dof_2_padded, 9, 35);

  // // Display the values in float converted to bf16 dof
  // std::cout << "\n-----------------------Padded dof1 col:--------------------------------\n";
  // printAsMatrix(l_dof_1_padded, 9, 35 + l_nz_idx.size());

  // std::cout << "\n-----------------------Padded dof2 col:--------------------------------\n";
  // printAsMatrix(l_dof_2_padded, 9, 35 + l_nz_idx.size());

  // First test
  float l_result[9][3][20] = { 0 };

  auto start = std::chrono::high_resolution_clock::now();

  bf16_gemm_one((libxsmm_bfloat16 *)l_dof_1_padded.data(),
                (libxsmm_bfloat16 *)l_stiff_padded.data(),
                (float *)l_result,
                3,
                9,
                20,
                35,
                l_nz_idx.size());

  // bf16_gemm_one((libxsmm_bfloat16 *)l_dof_2_padded.data(),
  //             (libxsmm_bfloat16 *)l_stiff_padded.data(),
  //             (float *)l_result,
  //             3,
  //             9,
  //             20,
  //             35,
  //             l_nz_idx.size());

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

  // Second test
  // float l_result[9][3][20] = { 0 };

  // auto start = std::chrono::high_resolution_clock::now();

  // bf16_gemm_one((libxsmm_bfloat16 *)l_dof_1_padded.data(),
  //               (libxsmm_bfloat16 *)l_stiff_padded.data(),
  //               (float *)l_result,
  //               9,
  //               3 * 20,
  //               35 + l_nz_idx.size());

  // bf16_gemm_one((libxsmm_bfloat16 *)l_dof_2_padded.data(),
  //               (libxsmm_bfloat16 *)l_stiff_padded.data(),
  //               (float *)l_result,
  //               9,
  //               3 * 20,
  //               35 + l_nz_idx.size());

  // auto end = std::chrono::high_resolution_clock::now();

  // std::chrono::duration<double> duration = end - start;
  // std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;


  // Third test
  // float l_result[9][3][20] = { 0 };

  // auto start = std::chrono::high_resolution_clock::now();

  // bf16_gemm_upconvert((libxsmm_bfloat16 *)l_dof_1_padded.data(),
  //                     (libxsmm_bfloat16 *)l_stiff_padded.data(),
  //                     (float *)l_result,
  //                     9,
  //                     3 * 20,
  //                     35 + l_nz_idx.size());

  // bf16_gemm_upconvert((libxsmm_bfloat16 *)l_dof_2_padded.data(),
  //                     (libxsmm_bfloat16 *)l_stiff_padded.data(),
  //                     (float *)l_result,
  //                     9,
  //                     3 * 20,
  //                     35 + l_nz_idx.size());

  // auto end = std::chrono::high_resolution_clock::now();

  // std::chrono::duration<double> duration = end - start;
  // std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

  std::cout << "\n-----------------------Result:--------------------------------\n";
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 20; ++k) {
        // int index = i * 9 * 20 + j * 20 + k;
        // std::cout << index << ": " ;
        std::cout << l_result[i][j][k] << "   ";
      }
      std::cout << std::endl;
    }
  }

  float l_reference[9][3][35] = { 0 };
  // auto start = std::chrono::high_resolution_clock::now();

  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    for( int64_t l_m = 0; l_m < 35; l_m++ ) {
      for( int64_t l_n = 0; l_n < 9; l_n++ ) {
        for( int64_t l_k = 0; l_k < 35; l_k++ ) {
          l_reference[l_di][l_n][l_m] +=  l_dofs[l_n][l_k] * stiff_test[l_di][l_k][l_m];
          int64_t index_stiff_test = l_di * 35 * 35 + l_k * 35 + l_m;
          int64_t index_l_dofs = l_n * 35 + l_k;
          
          // std::cout << index_l_dofs << "*" << index_stiff_test << " = " << l_dofs[l_n][l_k] << " * " << stiff_test[l_di][l_k][l_m] << std::endl;
        }
        int64_t index_l_reference = l_di * 9 * 35 * 35 + l_n * 35 + l_m * 35 * 9;
        // std::cout <<  " = " << index_l_reference << " = " << l_reference[l_di][l_n][l_m]  << std::endl;
      }
    }
  }
  //  auto end = std::chrono::high_resolution_clock::now();

  // std::chrono::duration<double> duration = end - start;
  // std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

  std::cout << "\n-----------------------Reference:--------------------------------" << std::endl;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 20; ++k) {
        // int index = i * 9 * 20 + j * 20 + k;
        // std::cout << index << ": " ;
        std::cout << l_reference[i][j][k] << "   ";
      }
       std::cout << std::endl;
    }
  }

  // 
  // std::cout << "\n-----------------------Reference:--------------------------------" << std::endl;

  // for (int i = 0; i < 3; ++i) {
  //   for (int j = 0; j < 9; ++j) {
  //     for (int k = 0; k < 35; ++k) {
  //       std::cout << l_reference[i][j][k] << "   ";
  //     }
  //      std::cout << std::endl;
  //   }
  // }

   float l_diff[9][3][20] = { 0 };
   std::cout << "\n-----------------------Difference:--------------------------------\n";
   for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 20; ++k) {
        // l_diff[i][j][k] = l_reference[i][j][k] - l_result[i][j][k];
        std::cout << l_reference[i][j][k] - l_result[i][j][k] << "   ";
      }
       std::cout << std::endl;
    }
  }

  delete[] l_dof_1;
  delete[] l_dof_2;
  delete[] l_stiff_1;
  delete[] l_stiff_2;

  return 0;
}