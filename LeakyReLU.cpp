#include "LeakyReLU.h"

std::vector<double> LeakyReLU::operator()(const std::vector<double> &z) const {
    std::vector<double> ret;
    for (auto &zi: z) {
        ret.push_back(zi >= 0 ? zi : zi * 0.01);
    }
    return ret;
}

std::vector<double> LeakyReLU::gradient(const std::vector<double> &dJ_da, const std::vector<double> &z, const std::vector<double> &a) const {
    int n = dJ_da.size();
    std::vector<double> ret(n);
    for (int i = 0; i < n; ++i) {
        ret[i] = z[i] >= 0 ? dJ_da[i] : dJ_da[i] * 0.01;
    }
    return ret;
}