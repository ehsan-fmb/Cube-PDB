#include <iostream>
#include <chrono>

class MicroTimer {
public:
    MicroTimer() : start(std::chrono::high_resolution_clock::now()), end(start) {
        // Constructor initializes start and end to the current time
    }

    // Method to start timing
    void startTimer() {
        start = std::chrono::high_resolution_clock::now();
    }

    // Method to stop timing
    void stopTimer() {
        end = std::chrono::high_resolution_clock::now();
    }

    // Method to get the duration in microseconds
    long long getDuration() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
};