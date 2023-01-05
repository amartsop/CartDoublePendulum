#pragma once 

#include <iostream>
#include <armadillo>

#include "CartPole.h"
#include "SingalProcessing.h"
#include "SystemLinearization.hpp"


class ControlSystem : public CartPole
{
    
public:
    ControlSystem();
    
    // Control signal
    arma::dvec control_signal(double t, const arma::dvec& x);

    // Get cart pole properties
    CartPoleProperties get_cart_pole_properties(void) {return m_cart_pole_props; }

    // System's equations 
    virtual arma::dvec f(double t, const arma::dvec& x, const arma::dvec& u);

private:

    // Desired state position
    arma::dvec m_xd = {2.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Desired state velocity
    arma::dvec m_xd_dot = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

private:
    // State equilibrium point
    arma::dvec m_x_star = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Control equilibrium point
    arma::dvec m_u_star = {0.0};

    // Model matrix A
    arma::dmat m_a_mat;

    // Control matrix B
    arma::dmat m_b_mat;

private:

        // Cart mass (kg)
        double m_m0 = 1.0;
        
        // Pendulum 1 mass (kg)
        double m_m1 = 2.0;
        
        // Pendulum 2 mass in (kg)
        double m_m2 = 2.0;

        // Pendulum 1 length (m)
        double m_l1 = 1.0;
        
        // Pendulum 2 length (m)
        double m_l2 = 1.0;

        // CartPole properties
        CartPoleProperties m_cart_pole_props;

        // CartPole constants     
        CartPoleConstants m_cpc;
};