#include "Linear.h"

std::vector<double> Linear::operator()(const std::vector<double> &z) const {
    return z;
}

std::vector<double> Linear::gradient(const std::vector<double> &dJ_da, const std::vector<double> &z, const std::vector<double> &a) const {
    return dJ_da;
}

