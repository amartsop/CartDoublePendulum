#include "../include/ControlSystem.h"

ControlSystem::ControlSystem()
{
    // Initialize cart poile properties struct
    m_cart_pole_props.cart_mass = m_m0;
    m_cart_pole_props.pendulum1_mass = m_m1;
    m_cart_pole_props.pendulum2_mass = m_m1;
    m_cart_pole_props.pendulum1_length = m_l1;
    m_cart_pole_props.pendulum2_length = m_l2;

    // Get cart pole constatns constants
    m_cpc = define_cart_pole_constants(m_cart_pole_props);

   // Get linear system
   SystemLinearization<ControlSystem> sys_lin(this, &ControlSystem::f, m_x_star,
        m_u_star);
    auto linear_system = sys_lin.get_linear_system();
    m_a_mat = linear_system.a_mat; 
    m_b_mat = linear_system.b_mat;
}

arma::dvec ControlSystem::control_signal(double t, const arma::dvec& x)
{
    arma::dmat k = {5.9404, -224.5153,  262.9463, 10.6674, -0.2585, 48.0373};
    arma::dvec u = -k * (x - m_xd);

    return u;
    // return {0};
}

// System's equations 
arma::dvec ControlSystem::f(double t, const arma::dvec& x, const arma::dvec& u)
{
    return (f_hat_fun(t, x, m_cpc) + b_fun(x, m_cpc) * u);
}