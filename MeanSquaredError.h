#ifndef MACHINE_LEARNING_MEANSQUAREDERROR_H
#define MACHINE_LEARNING_MEANSQUAREDERROR_H
#include "LossFunction.h"
#include <bits/stdc++.h>

struct MeanSquaredError : LossFunction {
    [[nodiscard]] double operator()(const std::vector<double> &y, const std::vector<double> &yhat) const override;

    [[nodiscard]] std::vector<double> gradient(const std::vector<double> &y, const std::vector<double> &yhat) const override;
};

#endif //MACHINE_LEARNING_MEANSQUAREDERROR_H
