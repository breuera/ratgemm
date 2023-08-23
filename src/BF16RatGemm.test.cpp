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

/* from https://github.com/libxsmm/libxsmm/blob/78695ed5b3d924a9434d335084e4d68234473294/src/libxsmm_lpflt_quant.c#L293 libxsmm_convert_bf16_f32 */
float upconvert_bf16_2(libxsmm_bfloat16 in) {
  /* up-convert is super simple */
  libxsmm_float_uint hybrid_in = { 0 };

  hybrid_in.u = in;
  /* DAZ */
  hybrid_in.u = ( (hybrid_in.u & 0x7f80) == 0x0 ) ? (unsigned short)(hybrid_in.u & 0x8000) : hybrid_in.u;
  hybrid_in.u = hybrid_in.u << 16;

  return hybrid_in.f;
}

void extract_submatrix(const libxsmm_bfloat16* l_stiff_padded,
                      int i_k, int i_n,
                      libxsmm_bfloat16* o_submatrix,
                      int i_start_col_idx) {
    const libxsmm_bfloat16* l_pointer = l_stiff_padded + i_start_col_idx;
    const int l_submatrix_width = i_n / 3;
    
    for (int row = 0; row < i_k; ++row) {
        for (int col = 0; col < l_submatrix_width; ++col) {
            o_submatrix[row * l_submatrix_width + col] = l_pointer[row * i_n + col];
        }
    }
}

/* from https://github.com/libxsmm/libxsmm/blob/c38c752f2d6dba92ffcbecc5f40d2bd652684d00/samples/equation/equation_matmul.c#L25 : gemm_bf16 */
void bf16_gemm_one(libxsmm_bfloat16* i_dof,
                   libxsmm_bfloat16* i_stiff,
                   float* o_result,
                   libxsmm_blasint i_di = 3, //3
                   libxsmm_blasint i_m = 9, //9
                   libxsmm_blasint i_n = 20, //20
                   libxsmm_blasint i_k = 35, //35
                   libxsmm_blasint i_nz_size = 10){ //10 in this case
   for( int64_t l_di = 0; l_di < i_di; l_di++ ) {
    for( int64_t l_m = 0; l_m < i_n; l_m++ ) {
      for( int64_t l_n = 0; l_n < i_m; l_n++ ) {
        float acc = 0.0f;
        for( int64_t l_k = 0; l_k < (i_k + i_nz_size); l_k++ ) {
          acc += upconvert_bf16_2(i_dof[l_n * (i_k + i_nz_size) + l_k]) * upconvert_bf16_2(i_stiff[l_di * i_n + l_k * i_n * i_di + l_m]);
        }
        o_result[l_di * i_m * i_n + l_n * i_n + l_m] += acc;
      }
    }
  }
}

void bf16_gemm_3x(libxsmm_bfloat16* i_dof,
                   libxsmm_bfloat16* i_stiff,
                   float* o_result,
                   libxsmm_blasint i_m,
                   libxsmm_blasint i_n,
                   libxsmm_blasint i_k){

  libxsmm_bfloat16* l_submatrix = new libxsmm_bfloat16[i_k * i_n];
  extract_submatrix(i_stiff, i_k, i_n*3, l_submatrix, 0 * i_n);

  for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
    for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
      float acc = 0.0f;
      for( int64_t l_k = 0; l_k < i_k; l_k++ ) {
        acc += upconvert_bf16_2(i_dof[(l_m * i_k )+ l_k]) * upconvert_bf16_2(l_submatrix[l_k * i_n + l_n]);
      }
      o_result[l_m * i_n + l_n + (0 * i_m * i_n)] += acc;
    }
  }

  extract_submatrix(i_stiff, i_k, i_n*3, l_submatrix,  1 * i_n);
  for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
    for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
      float acc = 0.0f;
      for( int64_t l_k = 0; l_k < i_k; l_k++ ) {
        acc += upconvert_bf16_2(i_dof[(l_m * i_k )+ l_k]) * upconvert_bf16_2(l_submatrix[l_k * i_n + l_n]);
      }
      o_result[l_m * i_n + l_n + (1 * i_m * i_n)] += acc;
    }
  }


  extract_submatrix(i_stiff, i_k, i_n*3, l_submatrix,  2 * i_n);
  for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
    for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
      float acc = 0.0f;
      for( int64_t l_k = 0; l_k < i_k; l_k++ ) {
        acc += upconvert_bf16_2(i_dof[(l_m * i_k )+ l_k]) * upconvert_bf16_2(l_submatrix[l_k * i_n + l_n]);
      }
      o_result[l_m * i_n + l_n + (2 * i_m * i_n)] += acc;
    }
  }

  delete[] l_submatrix;
}

void gemm_bf16_libxsmm(libxsmm_bfloat16* i_dof,
                       libxsmm_bfloat16* i_stiff,
                       float* o_result,
                       libxsmm_blasint i_m,
                       libxsmm_blasint i_n,
                       libxsmm_blasint i_k){

  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
                   l_flags |= LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI;
                   l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

  libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;

  libxsmm_gemmfunction l_bf16gemm = nullptr;

  // Strides
  const libxsmm_blasint lda = i_n;
  const libxsmm_blasint ldb = i_k;
  const libxsmm_blasint ldc = i_n;

  // Data types
  const libxsmm_datatype a_in_type = LIBXSMM_DATATYPE_BF16;
  const libxsmm_datatype b_in_type = LIBXSMM_DATATYPE_BF16;
  const libxsmm_datatype out_type = LIBXSMM_DATATYPE_F32;
  const libxsmm_datatype comp_type = LIBXSMM_DATATYPE_F32;

   libxsmm_gemm_shape l_shape = libxsmm_create_gemm_shape(i_n,
                                                          i_m,
                                                          i_k,
                                                          lda,
                                                          ldb,
                                                          ldc,
                                                          a_in_type,
                                                          b_in_type,
                                                          out_type,
                                                          comp_type);

  libxsmm_bfloat16* l_submatrix = new libxsmm_bfloat16[i_k * i_n];

  l_bf16gemm = libxsmm_dispatch_gemm_v2(l_shape,
                                        l_flags,
                                        l_prefetch_flags);

  libxsmm_gemm_param l_param;
  libxsmm_blasint i;
  const libxsmm_blasint d = 3;

  for (i = 0; i < d; ++i){
    extract_submatrix(i_stiff, i_k, i_n*d, l_submatrix, i_n*i);
    memset(&l_param,
            0,
            sizeof(libxsmm_gemm_param));

    l_param.a.primary = l_submatrix;
    l_param.b.primary = const_cast<libxsmm_bfloat16*>(i_dof);
    l_param.c.primary = o_result + (i * i_m * i_n);

    l_bf16gemm(&l_param);
  }
  
  delete[] l_submatrix;
}

void printMatrix(const libxsmm_bfloat16* matrix,
                 int rows,
                 int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      // Indexing formula for a 2D matrix: element at row i, column j is matrix[i * cols + j]
      std::cout << std::fixed << std::setw(10) << std::setprecision(7) << upconvert_bf16_2(matrix[i * cols + j]) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
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
      std::cout << upconvert_bf16_2(vec[i * i_n + j]) << " ";
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

void print_matrix(const char* label, const float* matrix, int rows, int cols) {
    std::cout << label << ":" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << std::setprecision(7) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void fill_matrix(float* matrix, int rows, int cols, float max_value) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = max_value * static_cast<float>(rand()) / RAND_MAX;
    }
}

void test_matrix_multiplication() {
  // Define the matrix dimensions
  const int i_m = 2;
  const int i_n = 4;
  const int i_k = 6;

  // Create input matrices
  float i_dof_f[i_m * i_k];
  float i_stiff_f[i_k * i_n * 3];
  libxsmm_bfloat16 i_dof[i_m * i_k];
  libxsmm_bfloat16 i_stiff[i_k * i_n * 3];
  libxsmm_bfloat16 i_dof_1[i_m * i_k];
  libxsmm_bfloat16 i_stiff_1[i_k * i_n * 3];
  float o_result_libxsmm[i_m * 3 * i_n] = {0};
  float o_result_manual[i_m * 3 * i_n] = {0};

  // Initialize i_dof and i_stiff with appropriate values
  fill_matrix(i_dof_f, i_m, i_k, 1.5);
  fill_matrix(i_stiff_f, i_k, i_n * 3, 1.5);
  convert_fp32_two_bf16(i_dof_f, i_dof, i_dof_1, i_m * i_k);
  convert_fp32_two_bf16(i_stiff_f, i_stiff, i_stiff_1, i_k * i_n * 3);

  // Call the functions
  gemm_bf16_libxsmm(i_dof, i_stiff_1, o_result_libxsmm, i_m, i_n, i_k);
  bf16_gemm_3x(i_dof, i_stiff_1, o_result_manual, i_m, i_n, i_k);

  // Compare results element-wise with a tolerance
  const float tolerance = 1e-5;
  bool mismatch = false;
  for (int i = 0; i < i_m * i_n; ++i) {
    if (std::abs(o_result_libxsmm[i] - o_result_manual[i]) > tolerance) {
      std::cout << "Mismatch at index " << i << ": "
                << "Libxsmm result = " << o_result_libxsmm[i]
                << ", Manual result = " << o_result_manual[i] << std::endl;
      mismatch = true;
    }
  }

  if (!mismatch) {
    std::cout << "Results match!" << std::endl;
  }

  printMatrix(i_dof, i_m, i_k);
  printMatrix(i_stiff_1, i_k, i_n * 3);

  // Print computed results
  print_matrix("Libxsmm Result Matrix", o_result_libxsmm, i_m * 3, i_n);
  print_matrix("Manual Result Matrix", o_result_manual, i_m * 3, i_n);
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
      // l_dofs[l_qt][l_md] = 1;
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
  //     std::cout << upconvert_bf16_2(l_stiff_1[i * (3*20) + j]) << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // std::cout << "\n-----------------------Converted to bf16 part two with many zeros:--------------------------------\n";
  // for (int i = 0; i < 35; i++) {
  //   for (int j = 0; j < 3*20; j++) {
  //     std::cout << upconvert_bf16_2(l_stiff_2[i * (3*20) + j]) << " ";
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
  float l_result_1[9][3][20] = { 0 };
  float l_result_1_2[9][3][20] = { 0 };

  // auto start = std::chrono::high_resolution_clock::now();
  const libxsmm_blasint m = 9;
  const libxsmm_blasint n = 20;
  const libxsmm_blasint k = 35 + l_nz_idx.size();

  // int start_col = 40; // Starting column offset
  // libxsmm_bfloat16* submatrix = new libxsmm_bfloat16[k * n];
  // extract_submatrix((const libxsmm_bfloat16*)l_stiff_padded.data(), k, n*3, submatrix, start_col);

  // std::cout << "\n-----------------------Converted to bf16:--------------------------------\n";
  // for (int i = 0; i < 35 + l_nz_idx.size(); i++) {
  //   for (int j = 0; j < 20; j++) {
  //     std::cout << upconvert_bf16_2(submatrix[i * (20) + j]) << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // delete[] submatrix;

  // std::cout << (const libxsmm_bfloat16*)l_stiff_padded.data() <<std::endl;
  // libxsmm_bfloat16 value = *((const libxsmm_bfloat16*)l_stiff_padded.data()+ 20)  ;
  // std::cout << "Value at the pointer: " << value << std::endl;

  auto start_time1 = std::chrono::high_resolution_clock::now();
  gemm_bf16_libxsmm((libxsmm_bfloat16*)l_dof_1_padded.data(),
                    (libxsmm_bfloat16*)l_stiff_padded.data(),
                    (float*)l_result_1,
                    m,
                    n,
                    k);
  gemm_bf16_libxsmm((libxsmm_bfloat16*)l_dof_2_padded.data(),
                   (libxsmm_bfloat16*)l_stiff_padded.data(),
                   (float*)l_result_1_2,
                   m,
                   n,
                   k);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 20; ++k) {
        l_result_1[i][j][k] += l_result_1_2[i][j][k];
      }
    }
  }
  auto end_time1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time1 = end_time1 - start_time1;

  std::cout << "\n-----------------------Result libxsmm:--------------------------------\n";
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 20; ++k) {
        // int index = i * 9 * 20 + j * 20 + k;
        // std::cout << index << ": " ;
        l_result_1[i][j][k] += l_result_1_2[i][j][k];
        std::cout << l_result_1[i][j][k] << "   ";
      }
      std::cout << std::endl;
    }
  }

  float l_result_3[9][3][20] = { 0 };
  auto start_time3 = std::chrono::high_resolution_clock::now();
  bf16_gemm_one((libxsmm_bfloat16 *)l_dof_1_padded.data(),
                (libxsmm_bfloat16 *)l_stiff_padded.data(),
                (float *)l_result_3,
                3,
                9,
                20,
                35,
                l_nz_idx.size());
  bf16_gemm_one((libxsmm_bfloat16 *)l_dof_2_padded.data(),
                (libxsmm_bfloat16 *)l_stiff_padded.data(),
                (float *)l_result_3,
                3,
                9,
                20,
                35,
                l_nz_idx.size());
  auto end_time3 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time3 = end_time3 - start_time3;

  std::cout << "\n-----------------------Result - 2x loop all:--------------------------------\n";
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 20; ++k) {
        // int index = i * 9 * 20 + j * 20 + k;
        // std::cout << index << ": " ;
        std::cout << l_result_3[i][j][k] << "   ";
      }
      std::cout << std::endl;
    }
  }

  float l_result_2[9][3][20] = { 0 };
  auto start_time2 = std::chrono::high_resolution_clock::now();
  bf16_gemm_3x((libxsmm_bfloat16 *)l_dof_1_padded.data(),
                (libxsmm_bfloat16 *)l_stiff_padded.data(),
                (float *)l_result_2,
                m,
                n,
                k);
  bf16_gemm_3x((libxsmm_bfloat16 *)l_dof_2_padded.data(),
                (libxsmm_bfloat16 *)l_stiff_padded.data(),
                (float *)l_result_2,
                m,
                n,
                k);
  auto end_time2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time2 = end_time2 - start_time2;


  // std::cout << "\n-----------------------Result - 3x submatrix -2x:--------------------------------\n";
  // for (int i = 0; i < 3; ++i) {
  //   for (int j = 0; j < 9; ++j) {
  //     for (int k = 0; k < 20; ++k) {
  //       // int index = i * 9 * 20 + j * 20 + k;
  //       // std::cout << index << ": " ;
  //       std::cout << l_result_2[i][j][k] << "   ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  float l_reference[9][3][35] = { 0 };
  auto start = std::chrono::high_resolution_clock::now();
  for( int64_t l_n = 0; l_n < 9; l_n++ ) {
    for( int64_t l_di = 0; l_di < 3; l_di++ ) {
      for( int64_t l_m = 0; l_m < 35; l_m++ ) {
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
   auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  // std::cout << "\n-----------------------Reference - Alex:--------------------------------" << std::endl;
  // for (int i = 0; i < 3; ++i) {
  //   for (int j = 0; j < 9; ++j) {
  //     for (int k = 0; k < 20; ++k) {
  //       // int index = i * 9 * 20 + j * 20 + k;
  //       // std::cout << index << ": " ;
  //       std::cout << l_reference[i][j][k] << "   ";
  //     }
  //      std::cout << std::endl;
  //   }
  // }

  // std::cout << "\n-----------------------Reference:--------------------------------" << std::endl;

  // for (int i = 0; i < 3; ++i) {
  //   for (int j = 0; j < 9; ++j) {
  //     for (int k = 0; k < 35; ++k) {
  //       std::cout << l_reference[i][j][k] << "   ";
  //     }
  //      std::cout << std::endl;
  //   }
  // }

  //  float l_diff[9][3][20] = { 0 };
  //  std::cout << "\n-----------------------Difference:--------------------------------\n";
  //  for (int i = 0; i < 3; ++i) {
  //   for (int j = 0; j < 9; ++j) {
  //     for (int k = 0; k < 20; ++k) {
  //       // l_diff[i][j][k] = l_reference[i][j][k] - l_result[i][j][k];
  //       std::cout << l_reference[i][j][k] - l_result[i][j][k] << "   ";
  //     }
  //      std::cout << std::endl;
  //   }
  // }

  std::cout << "\nExecution time of libxsmm function: " << std::fixed << std::setprecision(6) << elapsed_time1.count() << " seconds" << std::endl;
  std::cout << "Execution time of bf16 gemm loop function: " << std::fixed << std::setprecision(6) << elapsed_time3.count() << " seconds" << std::endl;
  std::cout << "Execution time of bf16 gemm 3x loop function: " << std::fixed << std::setprecision(6) << elapsed_time2.count() << " seconds" << std::endl;
  std::cout << "Execution time of reference loop function: " << std::fixed << std::setprecision(6) << duration.count() << " seconds" << std::endl;

  // test_matrix_multiplication();

  delete[] l_dof_1;
  delete[] l_dof_2;
  delete[] l_stiff_1;
  delete[] l_stiff_2;

  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    for( int64_t l_n = 0; l_n < 9; l_n++ ) {
      for( int64_t l_m = 0; l_m < 20; l_m++ ) {
        int64_t l_id_rat = l_di*20 + l_m;
        std::cout << l_result_2[l_n * m + l_id_rat] << "\t";
        REQUIRE( l_result_2[l_di][l_n][l_m] == Approx(l_result_1[l_n * m+ l_id_rat]) );
      }
    }
  }

}