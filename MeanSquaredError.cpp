#include "MeanSquaredError.h"

double MeanSquaredError::operator()(const vector<double> &y, const vector<double> &yhat) const {
    int m = y.size();
    auto diff = yhat - y;
    return dot(diff, diff) / 2 / m;
}

std::vector<double> MeanSquaredError::gradient(const vector<double> &y, const vector<double> &yhat) const {
    int m = y.size();
    return (yhat - y) / m;
}