#include "../include/EulerRotations.h"

// Basic rotation matrix wrt x axis
arma::dmat33 EulerRotations::basic_rotation_x(double x)
{
    double cx = cos(x); double sx = sin(x);
    return { {1.0f, 0.0f, 0.0f}, {0.0f, cx, -sx}, {0.0f, sx, cx}};
}

// Basic rotation matrix wrt y axis
arma::dmat33 EulerRotations::basic_rotation_y(double x)
{
    double cx = cos(x); double sx = sin(x);
    return {{cx, 0.0f, sx}, {0.0f, 1.0f, 0.0f}, {-sx, 0.0f, cx}};
}

// Basic rotation matrix wrt z axis
arma::dmat33 EulerRotations::basic_rotation_z(double x)
{
    double cx = cos(x); double sx = sin(x);
    return {{cx, -sx, 0.0f}, {sx, cx, 0.0f}, {0.0f, 0.0f, 1.0f}};
}

// Euler rotation matrix z-y'-x''
arma::dmat33 EulerRotations::rotation(double phi, double theta, double psi)
{
    return (basic_rotation_z(psi) * basic_rotation_y(theta) *
        basic_rotation_x(phi));
}

arma::dmat33 EulerRotations::rotation(const arma::dvec& euler_angles)
{
    return rotation(euler_angles(0), euler_angles(1), euler_angles(2));
}

// Connection of anglular velocity with euler angles derivative (w = G * theta_dot)
arma::dmat33 EulerRotations::G(double phi, double theta, double psi)
{
    return {{1.0, 0.0, -sin(theta)}, 
        {0.0, cos(phi), cos(theta) * sin(phi)}, 
        {0.0, -sin(phi), cos(phi) * cos(theta)}};
}

arma::dmat33 EulerRotations::G(const arma::dvec& euler_angles)
{
    return G(euler_angles(0), euler_angles(1), euler_angles(2));
}


arma::dmat33 EulerRotations::G_dot(const arma::dvec& euler_angles, const
    arma::dvec& euler_angles_dot)
{
    double phi = arma::as_scalar(euler_angles(0));
    double theta = arma::as_scalar(euler_angles(1));

    double phi_dot = arma::as_scalar(euler_angles_dot(0));
    double theta_dot = arma::as_scalar(euler_angles_dot(1));

    return {{0.0, 0.0, - cos(theta) * theta_dot},
        {0.0, - sin(phi) * phi_dot, cos(theta) * cos(phi) * phi_dot - 
        sin(theta) * sin(phi) * theta_dot},
        {0.0, - cos(phi) * phi_dot, -sin(phi) * cos(theta) * phi_dot - 
        cos(phi) * sin(theta) * theta_dot}};
}


// Rotation matrix to euler angles
arma::dvec EulerRotations::rotation_to_euler(const arma::dmat& rot_mat)
{
    // Rotation matrix components
    double r11 = rot_mat(0, 0);
    double r12 = rot_mat(0, 1);
    double r13 = rot_mat(0, 2);

    double r21 = rot_mat(1, 0);

    double r31 = rot_mat(2, 0);
    double r32 = rot_mat(2, 1);
    double r33 = rot_mat(2, 2);

    // Initialize euler angles
    double phi, theta, psi;

    if (r31 == 1 || r31 == -1) {
        // Set psi aribtrarily
        psi = 0;

        if(r31 == - 1) { theta = M_PI_2; phi = psi + atan2(r12, r13); }
        else {theta = - M_PI_2, phi = -psi + atan2(-r12, -r13); }
    }
    else {
        theta = - asin(r31);
        phi = atan2( (r32 / cos(theta)), (r33 / cos(theta)) );
        psi = atan2( (r21 / cos(theta)), (r11 / cos(theta)) );
    }

    return {phi, theta, psi};
}