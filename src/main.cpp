#include "Tracker.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
    int cameraIndex = 0;

    // allow user to pass camera index as argument
    // example: ./tracker 1
    if (argc > 1) {
        cameraIndex = std::stoi(argv[1]);
    }

    std::cout << "Starting Real-Time Object Tracker..." << std::endl;
    std::cout << "Press Q to quit" << std::endl;

    Tracker tracker;
    tracker.start(cameraIndex);

    std::cout << "Tracker stopped." << std::endl;
    return 0;
}