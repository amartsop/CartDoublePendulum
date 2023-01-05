#pragma once 

#include <iostream>
#include <vector>
#include <armadillo>
#include <numerical_solvers.hpp>
#include <numerical_jacobian.hpp>

template <class T>

class NumericalIntegration
{
public:
    // Consturctor
    NumericalIntegration(T* obj, arma::dvec (T::*TMemFun)(double t,
        const arma::dvec& x), arma::dmat (T::*TMemJac)(double t,
        const arma::dvec& x), double integration_step, uint state_vec_size);

    struct ODEParams{
        // Current state
        arma::dvec current_state;

        // Next time step 
        double next_time;

        // Function pointer
        arma::dvec (T::*MemFun)(double t, const arma::dvec& x);
    };

    struct ODEParamsJac{
        // Current state
        arma::dvec current_state;

        // Next time step 
        double next_time;

        // Function pointer
        arma::dmat (T::*MemFun)(double t, const arma::dvec& x);
    };

    struct ODERosenbrock{
        // Function pointer
        arma::dvec (T::*MemFun)(double t, const arma::dvec& x);
    };

    // Implicit euler method
    arma::dvec implicit_euler(double t, const arma::dvec& state);

    // Liniearly implicit euler method
    arma::dvec linearly_implicit_euler(double t, const arma::dvec& state);

private:
    // Objective and jacobian function for newton's algorithm implementation in 
    // implicit euler method 
    arma::dvec implicit_euler_obj_fun(const arma::dvec& state, ODEParams params);
    arma::dmat implicit_euler_obj_fun_jac(const arma::dvec& state, 
        ODEParamsJac params);

private:
    // Object handler (default copy constructor - no pointers)
    T* m_obj;

    // Data type of function pointer to be integrated
    typedef arma::dvec (T::*RMemFun)(double t, const arma::dvec& x);
    typedef arma::dmat (T::*RMemJac)(double t, const arma::dvec& x);

    // Function to be integrated
    RMemFun m_f;

    // Function jacobian
    RMemJac m_jac;

    // Integration step 
    double m_integration_step;

private:
    // Autonomous function g(y) for Rosenbrock methods
    arma::dvec autonomous_g(const arma::dvec& y);
};


template <class T>
NumericalIntegration<T>::NumericalIntegration(T* obj, arma::dvec
    (T::*TMemFun)(double t, const arma::dvec& x),
    arma::dmat (T::*TMemJac)(double t, const arma::dvec& x),
    double integration_step, uint state_vec_size)
{
    // Object handler
    m_obj = obj;

    // Pass function pointer to member variable
    m_f = TMemFun;

    // Pass function pointer to member variable
    m_jac = TMemJac;

    // Integration step 
    m_integration_step = integration_step;
}


/********************* Implicit euler integration scheme *********************/
template <class T>
arma::dvec NumericalIntegration<T>::implicit_euler(double t, const arma::dvec&
    state)
{
    /************************* Objective function **************************/
    // Initialize ode parameters for objective function
    ODEParams params;
    params.next_time = t + m_integration_step;
    params.current_state = state;
    params.MemFun = m_f;

    /******************* Objective function jacobian ********************/
    ODEParamsJac params_jac;
    params_jac.next_time = params.next_time;
    params_jac.current_state = state;
    params_jac.MemFun = m_jac;

    // Find nest state using newtons method (use current state as initial guess)
    return  ns::newton_raphson<NumericalIntegration, ODEParams, ODEParamsJac>
        (*this, &NumericalIntegration::implicit_euler_obj_fun,
        &NumericalIntegration::implicit_euler_obj_fun_jac, state,
        params, params_jac, 1.0e-4); 
}


// Implicit euler objective function
template <class T>
arma::dvec NumericalIntegration<T>::implicit_euler_obj_fun(const
    arma::dvec& state, ODEParams params)
{
    return (state - params.current_state -
        m_integration_step * (m_obj->*params.MemFun)(params.next_time, state));
}

// Implicit euler objective function
template <class T>
arma::dmat NumericalIntegration<T>::implicit_euler_obj_fun_jac(const
    arma::dvec& state, ODEParamsJac params)
{
    // Calculate jacobian
    auto jac = (m_obj->*params.MemFun)(params.next_time, state);

    return arma::eye(jac.n_rows, jac.n_cols) - m_integration_step * jac;
}


/***************** Linearly implicit euler integration scheme ******************/
// Implicit euler objective function
template <class T>
arma::dvec NumericalIntegration<T>::linearly_implicit_euler(double t,
    const arma::dvec& state)
{
    // Augmented state
    arma::dvec y = arma::join_vert(state, arma::dvec({t}));

    // Evaluation of g function
    arma::dvec gk = autonomous_g(y);

    // Evaluation of g jacobian
    arma::dmat jac = numjac::parallel_jac<NumericalIntegration>(*this, 
        &NumericalIntegration::autonomous_g, y);

    // Inversion component
    arma::dmat inv_com = arma::eye(jac.n_rows, jac.n_cols) -
        m_integration_step * jac;
    
    // Evaluate k1
    arma::dvec k1 = arma::solve(inv_com, m_integration_step * gk);

    // Update yk 
    y += k1;

    return y.rows(arma::span(0, y.n_rows - 2));
}


template <class T>
arma::dvec NumericalIntegration<T>::autonomous_g(const arma::dvec& y)

{
    arma::dvec x = y(arma::span(0, y.n_rows - 2));
    double t = arma::as_scalar(y(y.n_rows-1));
    return arma::join_vert((m_obj->*m_f)(t, x), arma::dvec({1.0}));
}