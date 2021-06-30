#include <chrono>
#include "common.h"

using namespace std;

my_timer_t start_clock() {
    return std::chrono::steady_clock::now();
}

double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

int pMod(int num, int denom) {
    return ((num % denom) + denom) % denom;
}

int divideAndRoundUp(int num, int denom) {
    if (num % denom > 0) {
        return num / denom + 1;
    }
    else {
        return num / denom;
    }
} 