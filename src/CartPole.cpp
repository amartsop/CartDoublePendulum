#include "../include/CartPole.h"

CartPole::CartPole()
{
    // Initialize cart poile properties struct
    m_cart_pole_props.cart_mass = m_m0;
    m_cart_pole_props.pendulum1_mass = m_m1;
    m_cart_pole_props.pendulum2_mass = m_m2;
    m_cart_pole_props.pendulum1_length = m_l1;
    m_cart_pole_props.pendulum2_length = m_l2;
    
    // Get cart pole constatns constants
    m_cpc = define_cart_pole_constants(m_cart_pole_props);
}

// System's equations 
arma::dvec CartPole::f(double t, const arma::dvec& x, const arma::dvec& u)
{
    return (f_hat_fun(t, x, m_cpc) + b_fun(x, m_cpc) * u);
}

// System f_hat function
arma::dvec CartPole::f_hat_fun(double t, const arma::dvec& x,
    const CartPoleConstants& params)
{
    // Split state vector
    arma::dvec x2 = x(arma::span(3, 5));

    // Get mass matrix
    arma::dmat d_mat = mass_matrix_fun(x, params);

    // Get system's Coriolis/centrifugal matrix function
    arma::dmat c_mat = coriolis_matrix_fun(x, params);

    // Get system's gravity vector fun
    arma::dvec g_vec = gravity_vector_fun(x, params);
    
    return arma::join_vert(x2, - arma::solve(d_mat, (c_mat * x2 + g_vec)));   
}

// System b function
arma::dvec CartPole::b_fun(const arma::dvec& x, const CartPoleConstants& params)
{
    // Split state vector
    arma::dvec x1 = x(arma::span(0, 2));

    // Get mass matrix
    arma::dmat d_mat = mass_matrix_fun(x, params);

    // Get system input vector
    arma::dvec h_vec = input_vector_fun(x, params);

    return arma::join_vert(arma::zeros<arma::dvec>(3), arma::solve(d_mat, h_vec));
}

// System mass matrix function
arma::dmat CartPole::mass_matrix_fun(const arma::dvec& x,
    const CartPoleConstants& params)
{
    // Initialize matrix
    arma::dmat d_mat = arma::zeros(3, 3);

    // Split state vector
    arma::dvec x1 = x(arma::span(0, 2));
    
    // Define matrix
    d_mat(0, 0) = params.d1;

    d_mat(0, 1) = params.d2 * cos(x1(1));
    d_mat(1, 0) = d_mat(0, 1);

    d_mat(0, 2) = params.d3 * cos(x1(2));
    d_mat(2, 0) = d_mat(0, 2);

    d_mat(1, 1) = params.d4;

    d_mat(1, 2) = params.d5 * cos(x1(1) - x1(2));
    d_mat(2, 1) = d_mat(1, 2);

    d_mat(2, 2) = params.d6;
    
    return d_mat;
}

// System Coriolis/centrifugal matrix function
arma::dmat CartPole::coriolis_matrix_fun(const arma::dvec& x,
    const CartPoleConstants& params)
{
    // Initialize matrix
    arma::dmat c_mat = arma::zeros(3, 3);

    // Split state vector
    arma::dvec x1 = x(arma::span(0, 2));
    arma::dvec x2 = x(arma::span(3, 5));

    // Define matrix
    c_mat(0, 1) = - params.d2 * sin(x1(1)) * x2(1);

    c_mat(0, 2) = - params.d3 * sin(x1(2)) * x2(2);

    c_mat(1, 2) = params.d5 * sin(x1(1) - x1(2)) * x2(2);

    c_mat(2, 1) = - params.d5 * sin(x1(1) - x1(2)) * x2(1);

    return c_mat;
}

// System gravity vector fun
arma::dvec CartPole::gravity_vector_fun(const arma::dvec& x,
    const CartPoleConstants& params)
{
    // Split state vector
    arma::dvec x1 = x(arma::span(0, 2));

    // Define vector
    return {0.0, - params.f1 * sin(x1(1)), - params.f2 * sin(x1(2))};
}

// Define cart pole constants
CartPole::CartPoleConstants CartPole::define_cart_pole_constants(
    const CartPole::CartPoleProperties& props)
{
    // Initialize cart pole constants
    CartPole::CartPoleConstants cpc;

    // Update d1
    cpc.d1 = props.cart_mass + props.pendulum1_mass + props.pendulum2_mass;

    // Update d2
    cpc.d2 = ( 0.5 * props.pendulum1_mass + props.pendulum2_mass ) *
        props.pendulum1_length;

    // Update d3
    cpc.d3 = 0.5 * props.pendulum2_mass * props.pendulum2_length;

    // Update d4 
    cpc.d4 = ( (1.0 / 3.0) * props.pendulum1_mass + props.pendulum2_mass ) *
        pow(props.pendulum1_length, 2.0);

    // Update d5 
    cpc.d5 = 0.5 * props.pendulum2_mass * props.pendulum1_length *
        props.pendulum2_length;
    
    // Update d6
    cpc.d6 = (1.0 / 3.0) * props.pendulum2_mass *
        pow(props.pendulum2_length, 2.0);

    // Update f1
    cpc.f1 = ( 0.5 * props.pendulum1_mass + props.pendulum2_mass)
        * props.pendulum1_length * props.g;

    // Update f2
    cpc.f2 = 0.5 *  props.pendulum2_mass * props.pendulum2_length * props.g;

    return cpc;
}

// Forward kinematics 
// Return the transformation matrices of the 
// three bodies with respect to their centre of mass. The frame convention 
// attaches the x axis along the longitudinal axis of each body
// the y axis along the lateral axis while the \ axis is pointing away from 
// the screen. In the case of pendulums an angle transformation is required.
std::vector<arma::dmat44> CartPole::forward_kinematics(const arma::dvec& state)
{
    // Split state vector
    arma::dvec x1 = state(arma::span(0, 2));

    // Initialize transformation matrix container 
    std::vector<arma::dmat44> transformation_container(3);

    // Cart transformation matrix
    transformation_container.at(0) = arma::eye(4, 4);
    transformation_container.at(0)(0, 3) = x1(0);

    // Pendulum 1 transformation matrix
    transformation_container.at(1) = arma::eye(4, 4);
    arma::dmat rot1 = EulerRotations::basic_rotation_z(M_PI/2.0 - x1(1));
    double pos1_x = x1(0) + m_l1/2.0 * sin(x1(1));
    double pos1_y = m_l1/2.0 * cos(x1(1));
    transformation_container.at(1)(arma::span(0, 2), arma::span(0, 2)) = rot1;
    transformation_container.at(1)(0, 3) = pos1_x;
    transformation_container.at(1)(1, 3) = pos1_y;
    
    // Pendulum 2 transformation matrix
    transformation_container.at(2) = arma::eye(4, 4);
    arma::dmat rot2 = EulerRotations::basic_rotation_z(M_PI/2.0 - x1(2));
    double pos2_x = x1(0) + m_l1 * sin(x1(1)) + m_l2/2.0 * sin(x1(2));
    double pos2_y = m_l1 * cos(x1(1)) + m_l2/2.0 * cos(x1(2));
    transformation_container.at(2)(arma::span(0, 2), arma::span(0, 2)) = rot2;
    transformation_container.at(2)(0, 3) = pos2_x;
    transformation_container.at(2)(1, 3) = pos2_y;

    return transformation_container;
}