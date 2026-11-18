#include "Linear.h"

std::vector<double> Linear::operator()(const std::vector<double> &z) const {
    return z;
}

std::vector<double> Linear::gradient(const std::vector<double> &dJ_da, const std::vector<double> &z, const std::vector<double> &a) const {
    return dJ_da;
}

void Linear::initialise(std::vector<std::vector<double> > &w, std::vector<double> &b, MyRandom &rng) const {
    double stddev = 2. / (w[0].size() + w.size());
    for (auto &row: w) {
        for (auto &wi: row) {
            wi = rng.normal(0, stddev);
        }
    }
}