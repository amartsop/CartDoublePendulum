#pragma once 

#include <iostream>
#include <thread>
#include <memory>

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/System.hpp>
#include <armadillo>

#include "CartPole.h"
#include "EulerRotations.h"


class CartPoleAnimation
{
public:
    CartPoleAnimation(const CartPole& cart_pole,
        const std::vector<double>& time_vec,
        const std::vector<arma::dvec>& state, double simulation_frequency);
    
    // Update 
    void animate(void);

    // Get widnow status
    bool get_window_status(void) { return m_window->isOpen(); }
    
    // Get window handle
    std::shared_ptr<sf::RenderWindow> get_window(void) { return m_window; }

public: 

    // Get aspect ratio
    double get_aspect_ratio(void);

    // Get scene width (pixels)
    double get_window_width(void) { return m_window_width; }

    // Get scene height (pixels)
    double get_window_height(void) { return m_window_height; }
    
    // Get scene width (meters)
    double get_window_width_in_meters(void) { return m_window_width_meters; }
    
    // Get scene height (meters)
    double get_window_height_in_meters(void) { return m_window_height_meters; }
    
private:

    // Window width
    double m_window_width = 1280.0;

    // Scene height
    double m_window_height = 720.0;

    // Aspect ratio
    double m_aspect_ratio = m_window_width / m_window_height;

    // Scene width (in meters)
    double m_window_width_meters, m_window_height_meters;

    // Pixels to meters 
    double m_pixels_to_meters_ratio;

    // Meters to pixels
    double meters_to_pixels(double meters);

    // Pixels to meters
    double pixels_to_meters(double pixels);

    // Define scene properties
    void define_scene_properties(void);

    // Animation frames per second
    double m_fps = 30;

    // Simulation frequency (Hz)
    double m_fs;
    
private:    

    // Window pointer
    std::shared_ptr<sf::RenderWindow> m_window;
    
    // Window name
    std::string m_window_name = "Cart pole";
    
private:
    // Clock
    sf::Clock m_clock;

private: 

    // Frame transformation (world frame fw wrt to scene frame fs)
    arma::dmat44 m_t_fw_fs = arma::eye(4, 4);

    // Cart pole handle 
    CartPole m_cart_pole;

    // Cart pole properties
    CartPole::CartPoleProperties m_cp_props;

    // World cart pole scale
    double m_cart_pole_scale = 3.0;

    // Convert to scene frame
    sf::Vector3f convert_to_scene_frame(const arma::dmat44& t_fi_fw);

    // State vector
    std::vector<arma::dvec> m_state_vec;

    // Time vector
    std::vector<double> m_time_vec;

    // Initialze cart pole bodies
    void initialize_cart_pole_bodies(void);

    // Render cart pole
    void render_cart_pole(const arma::dvec& state);

    // Cart pole shapes
    std::vector<std::shared_ptr<sf::RectangleShape>> m_cart_pole_shapes;
    
    // Cart pole bodies number
    const int m_cart_pole_bodies_num = 3;

    // Cart width to height ratio
    double m_cart_width_to_height_ratio = 2.0;

    // Pendulum width to height ratio
    double m_pendulum_width_to_height_ratio = 5.0;
};