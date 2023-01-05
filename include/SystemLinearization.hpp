#pragma once 

#include <iostream>
#include "numerical_jacobian.hpp"

template <class T>
class SystemLinearization
{

public:
    SystemLinearization(T* obj, arma::dvec (T::*TMemFun)(double t,
        const arma::dvec& x, const arma::dvec& u), const arma::dvec& x_star, 
        const arma::dvec& u_star);

    struct LinearSystem
    {
        // Systems model matrix
        arma::dmat a_mat;

        // Systems control matrix
        arma::dmat b_mat;
    
        // Equilibrium state point
        arma::dvec x_star;
    
        // Equilibrium control point
        arma::dvec u_star;
    };

    // Get linear system
    LinearSystem get_linear_system(void) { return m_linear_system; }

private:

    // Object handler (default copy constructor - no pointers)
    T* m_obj;

    // Data type of function pointer to be integrated
    typedef arma::dvec (T::*RMemFun)(double t, const arma::dvec& x,
        const arma::dvec& u);
    
    // Function to be linearized
    RMemFun m_f;

    // System function
    arma::dvec system_fun(const arma::dvec& x);

    // Control function
    arma::dvec control_fun(const arma::dvec& u);

    // Linear system
    LinearSystem m_linear_system;
};


template <class T>
SystemLinearization<T>::SystemLinearization(T* obj,
        arma::dvec (T::*TMemFun)(double t, const arma::dvec& x,
        const arma::dvec& u), const arma::dvec& x_star, 
        const arma::dvec& u_star) 
{
    // Object handler
    m_obj = obj;

    // Pass function pointer to member variable
    m_f = TMemFun;
    
    // Store equilibrium state
    m_linear_system.x_star = x_star;
    
    // Store equilibrium control
    m_linear_system.u_star = u_star;

    // Calculate systems model matrix
    m_linear_system.a_mat = numjac::parallel_jac<SystemLinearization>(*this,
        &SystemLinearization::system_fun, x_star);
    
    // Calculate systems control matrix
    m_linear_system.b_mat = numjac::parallel_jac<SystemLinearization>(*this,
        &SystemLinearization::control_fun, u_star);
}

// System function
template <class T>
arma::dvec SystemLinearization<T>::system_fun(const arma::dvec& x)
{
    return (m_obj->*m_f)(0.0, x, m_linear_system.u_star);
}

// Control function
template <class T>
arma::dvec SystemLinearization<T>::control_fun(const arma::dvec& u)
{
    return (m_obj->*m_f)(0.0, m_linear_system.x_star, u);
}