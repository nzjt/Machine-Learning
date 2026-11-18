#ifndef MACHINE_LEARNING_LAYERS_H
#define MACHINE_LEARNING_LAYERS_H
#include <bits/stdc++.h>

using namespace std;

struct Layer {
    int n, m;

    Layer(int n) : Layer(n, 1) {}

    Layer(int n, int m) : n(n), m(m) {}

    virtual void resize(int) {}

    [[nodiscard]] virtual int nParameters() const = 0;

    [[nodiscard]] virtual vector<double> predict(const vector<double> &activation) const = 0;

    template <class T>
    [[nodiscard]] vector<T> predict(const vector<T> &activations) const {
        vector<T> ret;
        for (const auto &t: activations) {
            ret.push_back(predict(t));
        }
        return ret;
    }

    [[nodiscard]] virtual vector<double> derivative(vector<double> &dJ_da, const vector<double> &z, const vector<double> &a) const = 0;

    virtual ~Layer() = default;
};

#endif //MACHINE_LEARNING_LAYERS_H
