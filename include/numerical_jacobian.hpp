#pragma once 

#include <iostream>
#include <armadillo>
#include <omp.h>

namespace numjac
{

// Numerical estimation of the jacobian of f(x) using central difference
template<class T>
arma::dmat central_difference(T& handle,
    arma::dvec (T::*TMemFun)(const arma::dvec& x), const arma::dvec& x,
        double tol=1.0e-5)
{
    // Set perturbation varibales
    arma::dvec x_pert_right = x;
    arma::dvec x_pert_left = x;

    // Evaluate function to get it's size
    arma::dvec f_right = (handle.*TMemFun)(x_pert_right);

    // Initialize jacobian
    arma::dmat jac = arma::zeros(f_right.n_rows, x.n_rows);

    // Estimation algorithm (central difference)
    for (uint i = 0; i < x.n_rows; ++i)
    {
        // Right and left perturbation
        x_pert_right(i) = x_pert_right(i) + tol;
        x_pert_left(i) = x_pert_left(i) - tol;

        // Central difference
        jac.col(i) = ((handle.*TMemFun)(x_pert_right) -
            (handle.*TMemFun)(x_pert_left)) / (2.0 * tol);

        // Update perturbations
        x_pert_left(i) = x(i);
        x_pert_right(i) = x(i);
    }
    return jac;
}

// Numerical estimation of the jacobian of f(t, x) using central difference
template<class T>
arma::dmat central_difference(T& handle,
        arma::dvec (T::*TMemFun)(double t, const arma::dvec& x),
        double t, const arma::dvec& x, double tol=1.0e-5)
{
    // Set perturbation varibales
    arma::dvec x_pert_right = x;
    arma::dvec x_pert_left = x;

    // Evaluate function to get it's size
    arma::dvec f_right = (handle.*TMemFun)(t, x_pert_right);

    // Initialize jacobian
    arma::dmat jac = arma::zeros(f_right.n_rows, x.n_rows);

    // Estimation algorithm (central difference)
    for (uint i = 0; i < x.n_rows; ++i)
    {
        // Right and left perturbation
        x_pert_right(i) = x_pert_right(i) + tol;
        x_pert_left(i) = x_pert_left(i) - tol;

        // Central difference
        jac.col(i) = ((handle.*TMemFun)(t, x_pert_right) -
            (handle.*TMemFun)(t, x_pert_left)) / (2.0 * tol);

        // Update perturbations
        x_pert_left(i) = x(i);
        x_pert_right(i) = x(i);
    }
    return jac;
}

// Numerical estimation of the jacobian of f(x) using forward difference
template<class T>
arma::dmat forward_difference(T& handle,
        arma::dvec (T::*TMemFun)(const arma::dvec& x), const arma::dvec& x,
        double tol=1.0e-5)
{
    arma::dvec fx = (handle.*TMemFun)(x); arma::dvec x_pert = x;
    arma::dmat jac = arma::zeros(fx.n_rows, x.n_rows);

    for (unsigned int i = 0; i < x.n_rows; ++i)
    {
        x_pert(i) = x_pert(i) + tol;
        jac.col(i) = ((handle.*TMemFun)(x_pert) - fx) / tol;
        x_pert(i) = x(i);
    }
    return jac;
}

// Numerical estimation of the jacobian of f(t, x) using forward difference
template<class T>
arma::dmat forward_difference(T& handle,
    arma::dvec (T::*TMemFun)(double t, const arma::dvec& x),
    double t, const arma::dvec& x, double tol=1.0e-5)
{
    arma::dvec fx = (handle.*TMemFun)(t, x); arma::dvec x_pert = x;
    arma::dmat jac = arma::zeros(fx.n_rows, x.n_rows);

    for (unsigned int i = 0; i < x.n_rows; i ++)
    {
        x_pert(i) = x_pert(i) + tol;
        jac.col(i) = ((handle.*TMemFun)(t, x_pert) - fx) / tol;
        x_pert(i) = x(i);
    }
    return jac;
}

// Numerical estimation of the jacobian of f(t, x) using forward difference
template<class T>
arma::dmat parallel_jac(T& handle,
    arma::dvec (T::*TMemFun)(double t, const arma::dvec& x), double t,
    const arma::dvec& x, double tol=1.0e-5)
{
    arma::dvec fx = (handle.*TMemFun)(t, x);
    arma::dmat jac = arma::zeros(fx.n_rows, x.n_rows);
    arma::dmat ej = arma::eye(x.n_rows, x.n_rows);


    #pragma omp parallel num_threads(4)
    {
        // Initialize local copy of jac
        arma::dmat jac_local = arma::zeros(fx.n_rows, x.n_rows);

        // Make a local copy of handle object
        T handle_cp = handle;

        #pragma omp for 
        for (unsigned int i = 0; i < x.n_rows; i++)
        {
            arma::dvec f_pert = (handle_cp.*TMemFun)(t, x + tol * ej.col(i));

            jac_local.col(i) = (f_pert - fx) / tol;
        }
        #pragma omp critical
        {
            jac += jac_local;
        }
    }
    return jac;
}


// Numerical estimation of the jacobian of f(x) using forward difference
template<class T>
arma::dmat parallel_jac(T& handle, arma::dvec (T::*TMemFun)(const arma::dvec& x),
    const arma::dvec& x, double tol=1.0e-5)
{
    arma::dvec fx = (handle.*TMemFun)(x);
    arma::dmat jac = arma::zeros(fx.n_rows, x.n_rows);
    arma::dmat ej = arma::eye(x.n_rows, x.n_rows);

    #pragma omp parallel num_threads(4)
    {
        // Initialize local copy of jac
        arma::dmat jac_local = arma::zeros(fx.n_rows, x.n_rows);

        // Make a local copy of handle object
        T handle_cp = handle;

        #pragma omp for 
        for (unsigned int i = 0; i < x.n_rows; i++)
        {
            arma::dvec f_pert = (handle_cp.*TMemFun)(x + tol * ej.col(i));

            jac_local.col(i) = (f_pert - fx) / tol;
        }
        #pragma omp critical
        {
            jac += jac_local;
        }
    }
    return jac;
}


// // Numerical estimation of the jacobian of f(x, params) using forward difference
// template<class T, class M>
// arma::dmat parallel_jac(T& handle,
//     arma::dvec (T::*TMemFun)(const arma::dvec& x, M params),
//     const arma::dvec& x, const M& params, double tol=1.0e-5)
// {
//     arma::dvec fx = (handle.*TMemFun)(x, params);
//     arma::dmat jac = arma::zeros(fx.n_rows, x.n_rows);
//     arma::dmat ej = arma::eye(x.n_rows, x.n_rows);

//     #pragma omp parallel num_threads(4)
//     {
//         // Initialize local copy of jac
//         arma::dmat jac_local = arma::zeros(fx.n_rows, x.n_rows);

//         // Make a local copy of handle object
//         T handle_cp = handle;

//         #pragma omp for 
//         for (unsigned int i = 0; i < x.n_rows; i++)
//         {
//             arma::dvec f_pert = (handle_cp.*TMemFun)(x + tol * ej.col(i), params);

//             jac_local.col(i) = (f_pert - fx) / tol;
//         }
//         #pragma omp critical
//         {
//             jac += jac_local;
//         }
//     }
//     return jac;
// }

}
