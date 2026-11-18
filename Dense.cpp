#include "ActivationFunction.h"
#include "Dense.h"
#include "linalg.h"
#include "MyRandom.h"

Dense::Dense(int n, ActivationFunction *activationFunction) : Layer(n), b(n), activationFunction(activationFunction) {}

void Dense::randomise(MyRandom &rng) {
    activationFunction->initialise(w, b, rng);
}

void Dense::read(ifstream &file) {
    uint32_t n, m;
    file.read((char*) &n, sizeof n);
    assert(n == this->n);
    file.read((char*) &m, sizeof m);
    assert(m == this->m);
    for (auto &row: w) {
        file.read((char*) row.data(), sizeof row[0] * m);
    }
    file.read((char*) b.data(), sizeof b[0] * n);
}

void Dense::write(ofstream &file) const {
    file.write((char*) &n, sizeof n);
    file.write((char*) &m, sizeof m);
    for (auto &row: w) {
        file.write((char*) row.data(), sizeof row[0] * m);
    }
    file.write((char*) b.data(), sizeof b[0] * n);
}

void Dense::resize(int m) {
    this->m = m;
    w.assign(n, vector<double>(m));
}

int Dense::nParameters() const {
    return n * (m + 1);
}

[[nodiscard]] vector<double> Dense::computeZ(const vector<double> &activations) const {
    vector<double> z = b;
    for (int i = 0; i < n; ++i) {
        double s = 0;
        auto wi = w[i].data();
#pragma omp simd reduction(+:s)
        for (int j = 0; j < m; ++j) {
            s += wi[j] * activations[j];
        }
        z[i] += s;
    }
    return z;
}

[[nodiscard]] vector<double> Dense::predict(const vector<double> &activations) const {
    return (*activationFunction)(computeZ(activations));
}

[[nodiscard]] vector<double> Dense::derivative(vector<double> &dJ_da, const vector<double> &z, const vector<double> &a) const {
    vector<double> dJ_dz = activationFunction->gradient(dJ_da, z, a);
    dJ_da.assign(m, 0);
    for (int k = 0; k < n; ++k) {
        double dJ_dz_k = dJ_dz[k];
        auto wk = w[k].data();
#pragma omp simd
        for (int j = 0; j < m; ++j) {
            dJ_da[j] += dJ_dz_k * wk[j];
        }
    }
    return dJ_dz;
}
