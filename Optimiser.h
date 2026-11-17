#ifndef MACHINE_LEARNING_OPTIMISER_H
#define MACHINE_LEARNING_OPTIMISER_H
#include "Dense.h"
#include <bits/stdc++.h>

struct Derivatives;

struct Optimiser {
    virtual void optimise(const Derivatives &derivatives, vector<Dense*> &layers) = 0;

    virtual ~Optimiser() = default;
};

#endif //MACHINE_LEARNING_OPTIMISER_H
