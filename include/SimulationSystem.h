#pragma once 

#include <iostream>

#include "numerical_jacobian.hpp"
#include "ControlSystem.h"
#include "CartPole.h"

class SimulationSystem
{

public:
    SimulationSystem();
    
    // Calculate system model function 
    arma::dvec f(double t, const arma::dvec& state_vector);

    // Calculate system's Jacobian
    arma::dmat dfdx(double t, const arma::dvec& x);
    
    // Get cart pole handle
    CartPole get_cart_pole_handle(void) { return m_cart_pole; }
    
private:

    // Cart pole 
    CartPole m_cart_pole;

    // Control system
    ControlSystem m_control_system;
};


