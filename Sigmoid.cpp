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
    return dJ_da * a * (1. - a);
}

void Sigmoid::initialise(std::vector<std::vector<double> > &w, std::vector<double> &b, MyRandom &rng) const {
    double stddev = 2. / (w[0].size() + w.size());
    for (auto &row: w) {
        for (auto &wi: row) {
            wi = rng.normal(0, stddev);
        }
    }
}