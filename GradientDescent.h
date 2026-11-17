#ifndef MACHINE_LEARNING_GRADIENTDESCENT_H
#define MACHINE_LEARNING_GRADIENTDESCENT_H
#include "Optimiser.h"
#include <bits/stdc++.h>

struct GradientDescent : Optimiser {
    double learningRate;

    GradientDescent(double learningRate) : learningRate(learningRate) {}

    void optimise(const Derivatives &derivatives, vector<Dense*> &layers) override;
};

#endif //MACHINE_LEARNING_GRADIENTDESCENT_H
