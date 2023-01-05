cmake_minimum_required( VERSION 3.10 )

project( simulation_setup )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Optimization parameters
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_FLAGS "-O2")

# Armadillo linking
find_package(Armadillo REQUIRED)

# SMFL linking
find_package(SFML COMPONENTS graphics network audio window system REQUIRED)

# Boost linking
find_package(Boost COMPONENTS program_options REQUIRED)

# Python linking
find_package(PythonLibs 2.7)

# OpenMP linking
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Include directories
include_directories(
    ./include
    )

set( SFML_Libraries
  sfml-graphics
  sfml-network
  sfml-audio
  sfml-window
  sfml-system
)

set(ALL_LIBS ${ARMADILLO_LIBRARIES} ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES} ${SFML_Libraries})

set(SOURCES 
  ./src/CartPole.cpp
  ./src/ControlSystem.cpp
  ./src/SimulationSystem.cpp
  ./src/CartPoleAnimation.cpp
  ./src/EulerRotations.cpp
  ./src/SignalProcessing.cpp
  )
  

# target
add_executable(main main.cpp ${SOURCES})

target_include_directories(main PRIVATE ${ARMADILLO_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${SFML_INCLUDE_DIRS})
    
target_link_libraries(main ${ALL_LIBS})