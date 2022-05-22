#include <chrono>
#include <iostream>

class Timer{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTimePoint;
    bool stopped;

public:
    Timer();

    void stop();
    
    ~Timer();
};