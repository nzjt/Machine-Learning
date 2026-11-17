#ifndef MACHINE_LEARNING_MYRANDOM_H
#define MACHINE_LEARNING_MYRANDOM_H
#include <bits/extc++.h>

struct MyRandom {
    __gnu_cxx::sfmt19937 rng;

    MyRandom() : rng(0) {}
    // MyRandom() : rng(chrono::high_resolution_clock::now().time_since_epoch().count()) {}

    double operator()(double l, double r) {
        return std::uniform_real_distribution<>(l, r)(rng);
    }
};

#endif //MACHINE_LEARNING_MYRANDOM_H
