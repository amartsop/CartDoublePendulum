#pragma once

#include <iostream>
#include <armadillo>


namespace ns
{
// Newton-Raphson method with known jacobian function f(x, params) 
template<class T, class M, class J>
arma::dvec newton_raphson(T& handle,
    arma::dvec (T::*TMemFun)(const arma::dvec& x, M params),
    arma::dmat (T::*TMemJac)(const arma::dvec& x, J params_jac),
    const arma::dvec& x0, const M& params, const J& params_jac,
    double tol=1.0e-4, double iter=10000) {

    /************** Newton raphson solution *************************/

    // Initialize    
    arma::dvec x = x0; arma::dvec x_prev = x0;

    // Evaluate function 
    arma::dvec g = (handle.*TMemFun)(x, params);

    // Declare jacobian
    arma::dmat jac;

    // Calculate residual
    double res = arma::norm(g);

    // Iteration counter 
    unsigned int iter_counter = 0;

    // Iteration flags 
    bool flag_residual = (res >= tol);
    bool flag_iteration = (iter_counter <= iter);
    bool flag_step = true;
    bool flag_total = (flag_residual || flag_step) && flag_iteration;

    // Absolute tolerance for step criterion
    double abs_tol = 1.0e-4; 

    while(flag_total)
    {
        // Calculate jacobian
        jac = (handle.*TMemJac)(x, params_jac);

        // Update solution
        x = x_prev - arma::solve(jac, g);

        // Update function
        g = (handle.*TMemFun)(x, params);

        // Update residual
        res = arma::norm(g);

        // Update counter
        iter_counter++;

        // Update stopping criteria
        flag_residual = (res >= tol);
        flag_iteration = (iter_counter <= iter);
        flag_step = (arma::norm(x - x_prev) >= abs_tol * (1.0 + arma::norm(x)));
        flag_total = (flag_residual || flag_step) && flag_iteration;

        // Update xprev
        x_prev = x;
    }

    if (!flag_iteration) {
        std::cerr << "Newton-Raphson method: No solution found";
        return x0;
    }
    else { return x; }
}

// Newton-Raphson method with known jacobian function f(x)
template<class T>
arma::dvec newton_raphson(T& handle,
    arma::dvec (T::*TMemFun)(const arma::dvec& x),
    arma::dmat (T::*TMemJac)(const arma::dvec& x),
    const arma::dvec& x0, double tol=1.0e-4, double iter=10000)
{
    /************** Newton raphson solution *************************/

    // Initialize    
    arma::dvec x = x0; arma::dvec x_prev = x0;

    // Evaluate function 
    arma::dvec g = (handle.*TMemFun)(x);

    // Declare jacobian
    arma::dmat jac;

    // Calculate residual
    double res = arma::norm(g);

    // Iteration counter 
    unsigned int iter_counter = 0;

    // Iteration flags 
    bool flag_residual = (res >= tol);
    bool flag_iteration = (iter_counter <= iter);
    bool flag_step = true;
    bool flag_total = (flag_residual || flag_step) && flag_iteration;

    // Absolute tolerance for step criterion
    double abs_tol = 1.0e-5; 

    while(flag_total)
    {
        // Calculate jacobian
        jac = (handle.*TMemJac)(x);

        // Update solution
        x = x_prev - arma::solve(jac, g);

        // Update function
        g = (handle.*TMemFun)(x);

        // Update residual
        res = arma::norm(g);

        // Update counter
        iter_counter++;

        // Update stopping criteria
        flag_residual = (res >= tol);
        flag_iteration = (iter_counter <= iter);
        flag_step = (arma::norm(x - x_prev) >= abs_tol * (1.0 + arma::norm(x)));
        flag_total = (flag_residual || flag_step) && flag_iteration;

        // Update xprev
        x_prev = x;
    }

    if (!flag_iteration) {
        std::cerr << "Newton-Raphson method: No solution found";
        return x0;
    }
    else { return x; }
}

}