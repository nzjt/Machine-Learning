#ifndef MACHINE_LEARNING_OPTIMISER_H
#define MACHINE_LEARNING_OPTIMISER_H
#include "Dense.h"
#include "Model.h"
#include <bits/stdc++.h>

struct Derivatives;

struct Optimiser {
    virtual void step(Model &model, const std::function<Derivatives(const Model&)> &computeDerivatives) = 0;

    void fit(const vector<vector<double>> &x, const vector<vector<double>> &y, int epochs, Model &model, MyRandom &rng, double sample);

    virtual ~Optimiser() = default;
};

#endif //MACHINE_LEARNING_OPTIMISER_H
