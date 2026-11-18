#include "ReLU.h"

std::vector<double> ReLU::operator()(const std::vector<double> &z) const {
    std::vector<double> ret;
    for (auto &zi: z) {
        ret.push_back(zi >= 0 ? zi : 0);
    }
    return ret;
}

std::vector<double> ReLU::gradient(const std::vector<double> &dJ_da, const std::vector<double> &z, const std::vector<double> &a) const {
    int n = dJ_da.size();
    std::vector<double> ret(n);
    for (int i = 0; i < n; ++i) {
        ret[i] = z[i] >= 0 ? dJ_da[i] : 0;
    }
    return ret;
}

void ReLU::initialise(std::vector<std::vector<double> > &w, std::vector<double> &b, MyRandom &rng) const {
    int m = w[0].size();
    double stddev = 2. / m;
    for (auto &row: w) {
        for (auto &wi: row) {
            wi = rng.normal(0, stddev);
        }
    }
    b.assign(b.size(), 0.01);
}