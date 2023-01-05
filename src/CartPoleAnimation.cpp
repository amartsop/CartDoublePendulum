#include "../include/CartPoleAnimation.h"

CartPoleAnimation::CartPoleAnimation(const CartPole& cart_pole, 
    const std::vector<double>& time_vec, const std::vector<arma::dvec>& state,
    double simulation_frequency)
{
    // Define window settings 
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;

    // Define render window    
    m_window = std::make_shared<sf::RenderWindow>(sf::VideoMode(m_window_width, 
        m_window_height), m_window_name, sf::Style::Default, settings);
    
    // Get cart pole handle
    m_cart_pole = cart_pole;

    // Get simulation frequency
    m_fs = simulation_frequency;
    
    // Get local copy of state 
    m_state_vec = state;

    // Get local copy of time
    m_time_vec = time_vec;

    // Define scene properties
    define_scene_properties();

    // Initialize cart pole bodies
    initialize_cart_pole_bodies();
}

// Animate
void CartPoleAnimation::animate(void)
{
    // Animation time
    double animation_period_sec = 1 / m_fps; // (s)
    uint animation_period =  1000 * animation_period_sec; // (ms)
    uint steps = m_fs / m_fps;
    
    // Restart clock
    m_clock.restart();

    // Get real time
    double real_time = 0;

    for (size_t i = 0; i < m_time_vec.size(); i = i + steps)    
    {
        if (m_window->isOpen())
        {
            // Clear window
            m_window->clear();

            // Check all the window's events that were triggered since the
            // last iteration of the loop
            sf::Event event;
            while (m_window->pollEvent(event))
            {
                // "close requested" event: we close the window
                if (event.type == sf::Event::Closed ||
                    sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                    m_window->close();
            }

            // Get elapsed time
            double elapsed_time = (double) m_clock.getElapsedTime().asSeconds();
        
            // Current state and time 
            arma::dvec x_current = m_state_vec.at(i);
            double t_current = m_time_vec.at(i);

            // Render rectangular bodies
            render_cart_pole(x_current);
        
            // Delay
            std::this_thread::sleep_for(std::chrono::milliseconds(animation_period));
        
            // Display
            m_window->display();
        }
        else { break; }
    }
}


// Render cart pole
void CartPoleAnimation::render_cart_pole(const arma::dvec& state)
{
    // Get transformation container 
    std::vector<arma::dmat44> transf_container = m_cart_pole.forward_kinematics(state);

    for (size_t i = 0; i < m_cart_pole_bodies_num; i++)
    {
        // Rectangle transformation with respect to world frame fw
        arma::dmat44 t_fi_fw = transf_container.at(i);
        
        // Get position and orientation of body with respect to scene frame F
        sf::Vector3f rect_pose = convert_to_scene_frame(t_fi_fw);
        
        // Convert position to pixels
        double rect_x = meters_to_pixels(rect_pose.x);
        double rect_y = meters_to_pixels(rect_pose.y);
        
        // Set its orientation
        m_cart_pole_shapes.at(i)->setRotation(rect_pose.z * 180.0 / M_PI);

        // Set its position
        m_cart_pole_shapes.at(i)->setPosition(sf::Vector2f(rect_x, rect_y));
    
        // Draw rectangle
        m_window->draw((*m_cart_pole_shapes.at(i)));
    }
}


// Convert to scene frame
sf::Vector3f CartPoleAnimation::convert_to_scene_frame(const arma::dmat44& t_fi_fw)
{
    // Transformation of body frame fi wrt to scene frame fs
    arma::dmat44 t_fi_fs = m_t_fw_fs * t_fi_fw;

    // Extract euler angles from transformation matrix
    // arma::dmat33 rot_fi_F = t_fi_fs.block(0, 0, 3, 3);
    arma::dmat33 rot_fi_F = t_fi_fs(arma::span(0, 2), arma::span(0, 2));
    
    // Get euler angles from rotation matrix
    arma::dvec euler = EulerRotations::rotation_to_euler(rot_fi_F);
    double euler_psi = euler(2);
    
    return sf::Vector3f(t_fi_fs(0, 3),  t_fi_fs(1, 3), euler_psi);
}

// Convert meter values to pixels
double CartPoleAnimation::meters_to_pixels(double meters)
{
    return m_pixels_to_meters_ratio * meters;
}

// Convert pixel values to meters
double CartPoleAnimation::pixels_to_meters(double pixels)
{
    return pixels / m_pixels_to_meters_ratio;
}

// Define scene properties
void CartPoleAnimation::define_scene_properties(void)
{
    // Get cart pole properties 
    m_cp_props = m_cart_pole.get_cart_pole_properties();
   
    // Define world dimensions based on cart pole properties
    m_window_height_meters = m_cart_pole_scale *
        (m_cp_props.pendulum1_length + m_cp_props.pendulum2_length);

    m_window_width_meters = m_window_height_meters * m_aspect_ratio;

    m_pixels_to_meters_ratio = m_window_width / m_window_width_meters;

    // Define scene transform
    m_t_fw_fs(1, 1) = -1.0;
    m_t_fw_fs(2, 2) = -1.0;
    m_t_fw_fs(0, 3) = m_window_width_meters / 2.0;
    m_t_fw_fs(1, 3) = m_window_height_meters / 2.0;
}

// Initialze cart pole bodies
void CartPoleAnimation::initialize_cart_pole_bodies(void)
{
    // Initialize cart pole shapes
    m_cart_pole_shapes.resize(m_cart_pole_bodies_num);

    // Initialize cart 
    double cart_width = meters_to_pixels(m_cp_props.pendulum1_length);
    double cart_height = cart_width / m_cart_width_to_height_ratio;

    // Define cart 
    m_cart_pole_shapes.at(0) = std::make_shared<sf::RectangleShape>(
        sf::Vector2f(cart_width, cart_height));

    m_cart_pole_shapes.at(0)->setOrigin(sf::Vector2f(cart_width/2.0,
        cart_height/2.0));
        
    m_cart_pole_shapes.at(0)->setFillColor(sf::Color::Red);

    // Initialize pendulum 1
    double pendulum1_width = meters_to_pixels(m_cp_props.pendulum1_length);
    double  pendulum1_height = pendulum1_width / m_pendulum_width_to_height_ratio;
    
    m_cart_pole_shapes.at(1) = std::make_shared<sf::RectangleShape>(
        sf::Vector2f(pendulum1_width, pendulum1_height));

    m_cart_pole_shapes.at(1)->setOrigin(sf::Vector2f(pendulum1_width/2.0,
        pendulum1_height/2.0));

    m_cart_pole_shapes.at(1)->setFillColor(sf::Color::Blue);

    // Initialize pendulum 2
    double pendulum2_width = meters_to_pixels(m_cp_props.pendulum2_length);
    double  pendulum2_height = pendulum2_width / m_pendulum_width_to_height_ratio;

    m_cart_pole_shapes.at(2) = std::make_shared<sf::RectangleShape>(
        sf::Vector2f(pendulum2_width, pendulum2_height));

    m_cart_pole_shapes.at(2)->setOrigin(sf::Vector2f(pendulum2_width/2.0,
        pendulum2_height/2.0));

    m_cart_pole_shapes.at(2)->setFillColor(sf::Color::Magenta);
}