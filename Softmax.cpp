#include "linalg.h"
#include "Softmax.h"

std::vector<double> Softmax::operator()(const std::vector<double> &z) const {
    std::vector<double> e = exp(z);
    return e / sum(e);
}

std::vector<double> Softmax::gradient(const std::vector<double> &dJ_da, const std::vector<double> &z, const std::vector<double> &a) const {
    int n = dJ_da.size();
    std::vector<double> ret(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ret[i] += dJ_da[j] * a[j] * ((i == j) - a[i]);
        }
    }
    return ret;
}

void Softmax::initialise(std::vector<std::vector<double> > &w, std::vector<double> &b, MyRandom &rng) const {
    double stddev = 2. / (w[0].size() + w.size());
    for (auto &row: w) {
        for (auto &wi: row) {
            wi = rng.normal(0, stddev);
        }
    }
}