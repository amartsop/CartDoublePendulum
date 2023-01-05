#pragma once 

#include <iostream>

class SingalProcessing
{
public:
    SingalProcessing() {};

    /**
    * ======================================================================
    * @brief Saturates a given signal
    *
    * @param x is the signal to be saturated.
    * @param z is the limit of saturation.
    * @retval y is the saturated signal.
    * ======================================================================
    */
    inline static double saturation(double x, double z){
        if (x < z) { return -1; }
        else if (x > z) {return 1; }
        else {return x/z; }
    }
};




