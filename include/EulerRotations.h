#pragma once 

#include <iostream>
#include <armadillo>
#include <math.h>
#include <omp.h>

class EulerRotations
{
public:
    // Basic rotation matrix wrt x axis
    static arma::dmat33 basic_rotation_x(double x);
    
    // Basic rotation matrix wrt y axis
    static arma::dmat33 basic_rotation_y(double x);

    // Basic rotation matrix wrt z axis
    static arma::dmat33 basic_rotation_z(double x);

    // Euler rotation matrix z-y'-x''
    static arma::dmat33 rotation(double phi, double theta, double psi);
    static arma::dmat33 rotation(const arma::dvec& euler_angles);

    // Connection of anglular velocity with euler angles derivative (w = G * theta_dot)
    static arma::dmat33 G(double phi, double theta, double psi);
    static arma::dmat33 G(const arma::dvec& euler_angles);

    // Time derivative of G matrix
    static arma::dmat33 G_dot(const arma::dvec& euler_angles, const arma::dvec& 
        euler_angles_dot);

    // Rotation matrix to euler angles
    static arma::dvec rotation_to_euler(const arma::dmat& rot_mat);
};
