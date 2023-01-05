#pragma once

#include <iostream>
#include <armadillo>

#include "EulerRotations.h"

class CartPole
{
    
public:
    CartPole();

    // System's equations 
    virtual arma::dvec f(double t, const arma::dvec& x, const arma::dvec& u);

    struct CartPoleProperties
    {
        // Cart mass (kg)
        double cart_mass;        
        
        // Pendulum 1 mass (kg)
        double pendulum1_mass;
        
        // Pendulum 2 mass (kg)
        double pendulum2_mass;

        // Pendulum 1 length (m)
        double pendulum1_length;
        
        // Pendulum 2 length (m)
        double pendulum2_length;
    
        // Gravity constant (m / s^2)
        double g = 9.80665;
    };
    
    struct CartPoleConstants
    {
        // Constant d1
        double d1;

        // Constant d2
        double d2;
    
        // Constant d3
        double d3;

        // Constant d4
        double d4;

        // Constant d5
        double d5;

        // Constant d6
        double d6;
    
        // Constant f1
        double f1;

        // Constant f2
        double f2;
    };

    // Get cart pole properties
    virtual CartPoleProperties get_cart_pole_properties(void) {return m_cart_pole_props; }

    // Define cart pole constants
    CartPoleConstants define_cart_pole_constants(const CartPoleProperties& props);

    // Forward kinematics 
    // Return the transformation matrices of the 
    // three bodies with respect to their centre of mass. The frame convention 
    // attaches the x axis along the longitudinal axis of each body
    // the y axis along the lateral axis while the \ axis is pointing away from 
    // the screen. In the case of pendulums an angle transformation is required.
    std::vector<arma::dmat44> forward_kinematics(const arma::dvec& state);

protected:

    // System f_hat function
    static arma::dvec f_hat_fun(double t, const arma::dvec& x,
        const CartPoleConstants& params);

    // System b function
    static arma::dvec b_fun(const arma::dvec& x, const CartPoleConstants& params);

    // System mass matrix function
    static arma::dmat mass_matrix_fun(const arma::dvec& x,
        const CartPoleConstants& params);

    // System Coriolis/centrifugal matrix function
    static arma::dmat coriolis_matrix_fun(const arma::dvec& x,
        const CartPoleConstants& params);

    // System gravity vector fun
    static arma::dvec gravity_vector_fun(const arma::dvec& x,
        const CartPoleConstants& params);
    
    // System input vector 
    inline static arma::dvec input_vector_fun(const arma::dvec& x,
        const CartPoleConstants& params) { return {1.0, 0.0, 0.0}; }

protected:

    // System dimensions
    int m_n = 6;

    // Dimensions of theta vector
    int m_n_theta = 3;

    // Dimensions of control signal
    int m_n_u = 1;

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