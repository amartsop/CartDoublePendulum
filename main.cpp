#include <iostream>
#include <armadillo>
#include <omp.h>

#include "./include/SimulationSystem.h"
#include "./include/CartPoleAnimation.h"
#include "./include/numerical_integration.hpp"

int main(void)
{
    // Timing
    double t_init = 0.0;
    double t_final = 40.0; // Final time (s)
    double fs = 1.0e3;  // Simulation frequency (Hz)

    // Integration step
    double integr_step = 1.0 / fs;  

    // Define time vector
    std::vector<double> time_vector;
    time_vector.push_back(t_init);

    // Initialize state
    arma::dvec x_init = {0.0, 0.5, 0.4, 0.0, 0.0, 0.0};

    // Define state vector
    std::vector<arma::dvec> state_vector;
    state_vector.push_back(x_init);

    // Initialize total system
    SimulationSystem sim_system;

    /************************ Simulation *************************/

    // Iteration counter 
    uint counter = 0;

    // Initialize time
    double t = time_vector.at(0);

    // Problem solver 
    NumericalIntegration<SimulationSystem> ni(&sim_system,
        &SimulationSystem::f, &SimulationSystem::dfdx, integr_step,
        state_vector.at(0).n_rows);

    // Initialize execution timer
    double initial_time = omp_get_wtime();

    while (t <= t_final)
    {
        // System solution 
        arma::dvec x = ni.linearly_implicit_euler(t, state_vector.at(counter));
        state_vector.push_back(x);

        // Update time and counter
        t += integr_step; counter += 1;

        // Update time vector 
        time_vector.push_back(t);
    }

    // Simulation time 
    double simulation_time = omp_get_wtime() - initial_time;

    // Define animation
    CartPoleAnimation animation(sim_system.get_cart_pole_handle(), 
        time_vector, state_vector, fs);

    // Animation
    animation.animate();
} 

