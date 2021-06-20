#include <chrono>

using namespace std;

timer_t tart_clock() {
    return std::chrono::steady_clock::now();
}

double stop_clock_and_add(timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

int pMod(int num, int denom) {
    return ((num % denom) + denom) % denom;
}
    