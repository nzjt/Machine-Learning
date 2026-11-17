#include "Sigmoid.h"
#include "linalg.h"

std::vector<double> Sigmoid::operator()(const std::vector<double> &z) const {
    std::vector<double> ret;
    for (auto &zi: z) {
        ret.push_back(1 / (1 + exp(-zi)));
    }
    return ret;
}

std::vector<double> Sigmoid::gradient(const std::vector<double> &dJ_da, const std::vector<double> &z, const std::vector<double> &a) const {
    return dJ_da * a * a * exp(-z);
}