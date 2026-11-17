#ifndef MACHINE_LEARNING_TRANSFORM_H
#define MACHINE_LEARNING_TRANSFORM_H
#include <bits/stdc++.h>

using namespace std;

struct Transform {
    struct Parameters {
        double mean = 0, range = 1;
    };
    vector<Parameters> params;

    Transform(int n) : params(n) {}

    void fit(const vector<double> &x) {
        vector<vector<double>> x2(x.size());
        for (int i = 0; i < x.size(); ++i) {
            x2[i] = {x[i]};
        }
        fit(x2);
    }

    void fit(const vector<vector<double>> &x) {
        int m = x.size(), n = x[0].size();
        for (int i = 0; i < n; ++i) {
            double mn = x[0][i], mx = x[0][i], sum = x[0][i];
            for (int j = 0; ++j < m;) {
                mn = min(mn, x[j][i]);
                mx = max(mx, x[j][i]);
                sum += x[j][i];
            }
            params[i] = {sum / m, mn == mx ? 1 : mx - mn};
        }
    }

    [[nodiscard]] vector<double> operator()(const vector<double> &x) const {
        int n = x.size();
        vector<double> ret(n);
        for (int i = 0; i < n; ++i) {
            ret[i] = (x[i] - params[i].mean) / params[i].range;
        }
        return ret;
    }

    template <class T>
    [[nodiscard]] vector<T> operator()(const vector<T> &x) const {
        vector<T> ret;
        for (int i = 0; i < x.size(); ++i) {
            ret.push_back((*this)(x[i]));
        }
        return ret;
    }

    [[nodiscard]] vector<double> inverse(const vector<double> &x) const {
        int n = x.size();
        vector<double> ret(n);
        for (int i = 0; i < n; ++i) {
            ret[i] = x[i] * params[i].range + params[i].mean;
        }
        return ret;
    }

    template <class T>
    [[nodiscard]] vector<T> inverse(const vector<T> &x) const {
        vector<T> ret;
        for (int i = 0; i < x.size(); ++i) {
            ret.push_back(inverse(x[i]));
        }
        return ret;
    }
};

#endif //MACHINE_LEARNING_TRANSFORM_H
