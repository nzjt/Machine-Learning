#ifndef MACHINE_LEARNING_LINEAR_H
#define MACHINE_LEARNING_LINEAR_H
#include "ActivationFunction.h"
#include <bits/stdc++.h>

struct Linear : ActivationFunction {
    using ActivationFunction::operator();

    [[nodiscard]] std::vector<double> operator()(const std::vector<double> &z) const override;

    using ActivationFunction::gradient;

    [[nodiscard]] std::vector<double> gradient(const std::vector<double> &dJ_da, const std::vector<double> &z, const std::vector<double> &a) const override;
};

#endif //MACHINE_LEARNING_LINEAR_H
