#include <chrono>

using namespace std;

chrono::time_point<std::chrono::steady_clock> start_clock() {
    return std::chrono::steady_clock::now();
}

void stop_clock_and_add(chrono::time_point<std::chrono::steady_clock> &start, double* timer) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    *timer += diff.count();
}


int pMod(int num, int denom) {
    return ((num % denom) + denom) % denom;
}