#include "../include/SimulationSystem.h"

SimulationSystem::SimulationSystem(void)
{
    
    
}

// Calculate system model function 
arma::dvec SimulationSystem::f(double t, const arma::dvec& state_vector)
{
    // // Calculate control signal
    // arma::dvec u = m_control_system.control_signal(t, state_vector);
    arma::dvec u = arma::zeros(1, 1);

    // Calculate system model function
    return m_cart_pole.f(t, state_vector, u);
}

// Calculate system's Jacobian
arma::dmat SimulationSystem::dfdx(double t, const arma::dvec& x)
{
    return numjac::parallel_jac<SimulationSystem>(*this, &SimulationSystem::f, t, x);
}