#include "timer.h"


Timer::Timer(){
    stopped = false;
    this->m_startTimePoint = std::chrono::high_resolution_clock::now();
}

void Timer::stop(){
    auto endTimePoint = std::chrono::high_resolution_clock::now();

    auto start        = std::chrono::time_point_cast<std::chrono::microseconds>(m_startTimePoint).time_since_epoch().count();
    auto end          = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch().count();

    auto duration = end - start;
    std::cout << duration << std::endl;
    stopped = true;
}

    
Timer::~Timer(){
    if(!stopped) stop();
}
