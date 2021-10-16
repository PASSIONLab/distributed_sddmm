#pragma once

#include "SpmatLocal.hpp"
#include <string>

#define MINIMUM_BENCH_TIME 10.0

void benchmark_algorithm(SpmatLocal* S, 
        string algorithm_name,
        string output_file,
        bool fused,
        int R,
        int c,
        string app 
        );