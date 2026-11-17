#ifndef MACHINE_LEARNING_LOSSES_H
#define MACHINE_LEARNING_LOSSES_H
#include "linalg.h"
#include <bits/stdc++.h>

using namespace std;

struct LossFunction {
    [[nodiscard]] virtual double operator()(const vector<double> &y, const vector<double> &yhat) const = 0;

    template <class T>
    double operator()(const vector<T> &y, const vector<T> &yhat) const {
        int m = y.size();
        double loss = 0;
        for (int i = 0; i < m; ++i) {
            loss += (*this)(y[i], yhat[i]);
        }
        return loss / m;
    }

    [[nodiscard]] virtual vector<double> gradient(const vector<double> &y, const vector<double> &yhat) const = 0;

    template <class T>
    vector<double> gradient(const vector<T> &y, const vector<T> &yhat) const {
        int m = yhat.size();
        vector<double> ret;
        for (int i = 0; i < m; ++i) {
            ret += gradient(y[i], yhat[i]);
        }
        return ret / m;
    }

    virtual ~LossFunction() = default;
};

#endif //MACHINE_LEARNING_LOSSES_H
