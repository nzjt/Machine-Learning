#include "Model.h"

Model::Model(int inputSize, const vector<Dense*> &layers) : layers(layers) {
    this->layers[0]->resize(inputSize);
    for (int i = 1; i < layers.size(); ++i) {
        this->layers[i]->resize(layers[i - 1]->n);
    }
}

void Model::setLossFunction(LossFunction *lossFunction) {
    this->lossFunction = lossFunction;
}

void Model::setRegularisation(double regularisation) {
    this->regularisation = regularisation;
}

void Model::randomise(MyRandom &rng) {
    for (auto &layer: layers) {
        layer->randomise(rng);
    }
}

void Model::read(ifstream &file) {
    uint32_t k;
    file.read((char*) &k, sizeof k);
    assert(k == layers.size());
    for (auto layer: layers) {
        layer->read(file);
    }
}

void Model::write(ofstream &file) const {
    uint32_t k = layers.size();
    file.write((char*) &k, sizeof k);
    for (auto layer: layers) {
        layer->write(file);
    }
}

[[nodiscard]] vector<double> Model::predict(const vector<double> &x) const {
    auto activations = x;
    for (const auto &layer : layers) {
        activations = layer->predict(activations);
    }
    return activations;
}

int Model::nParameters() const {
    int total = 0;
    for (const auto &layer : layers) {
        total += layer->nParameters();
    }
    return total;
}

void Model::computeDerivatives(const vector<double> &x, const vector<double> &y, Derivatives &accumulator) const {
    struct Cache {
        vector<double> z, a;
    };
    int K = layers.size();
    vector<Cache> cache(K);
    auto activations = x;
    for (int i = 0; i < K; ++i) {
        cache[i].z = layers[i]->computeZ(activations);
        cache[i].a = (*layers[i]->activationFunction)(cache[i].z);
        activations = cache[i].a;
    }
    vector<double> dJ_da = lossFunction->gradient(y, activations);
    auto &[dJ_dw, dJ_db] = accumulator;
    for (int i = K; i--;) {
        int n = layers[i]->n, m = layers[i]->w[0].size();
        vector<double> dJ_dz = layers[i]->derivative(dJ_da, cache[i].z, cache[i].a);
        const vector<double> &a = i ? cache[i - 1].a : x;
        for (int j = 0; j < n; ++j) {
            auto &rowW = dJ_dw[i][j];
            double dJ_dz_j = dJ_dz[j];
#pragma omp simd
            for (int k = 0; k < m; ++k) {
                rowW[k] += a[k] * dJ_dz_j;
            }
        }
        dJ_db[i] += dJ_dz;
    }
}

Model::Parameters Model::computeDerivatives(int m, const vector<vector<double>> &x, const vector<vector<double>> &y, const vector<bool> &doSample) const {
    int M = x.size(), K = layers.size();
    Derivatives gradient;
    gradient.dJ_dw.resize(K);
    gradient.dJ_db.resize(K);
    for (int i = 0; i < K; ++i) {
        gradient.dJ_dw[i] = layers[i]->w * regularisation;
        gradient.dJ_db[i] = layers[i]->b * regularisation;
    }
    Derivatives empty;
    empty.dJ_dw.resize(K);
    empty.dJ_db.resize(K);
    for (int i = 0; i < K; ++i) {
        empty.dJ_dw[i] = vector<vector<double>>(layers[i]->w.size(), vector<double>(layers[i]->w[0].size()));
        empty.dJ_db[i] = vector<double>(layers[i]->b.size());
    }
#pragma omp parallel
    {
        Derivatives local = empty;
#pragma omp for nowait
        for (int i = 0; i < M; ++i) {
            if (doSample[i]) {
                computeDerivatives(x[i], y[i], local);
            }
        }
#pragma omp critical
        gradient.dJ_dw += local.dJ_dw;
        gradient.dJ_db += local.dJ_db;
    }
    Parameters ret(nParameters());
    auto ptr = ret.data();
    for (int i = 0; i < K; ++i) {
        for (const auto &row : gradient.dJ_dw[i]) {
            memcpy(ptr, row.data(), row.size() * sizeof(double));
            ptr += row.size();
        }
        auto &dJ_db = gradient.dJ_db[i];
        memcpy(ptr, dJ_db.data(), dJ_db.size() * sizeof(double));
        ptr += dJ_db.size();
    }
    return ret / m;
}

void Model::operator+=(const Parameters &parameters) {
    auto ptr = parameters.data();
    for (auto &layer : layers) {
        for (auto &row : layer->w) {
            for (auto &wi: row) {
                wi += *ptr++;
            }
        }
        for (auto &bi: layer->b) {
            bi += *ptr++;
        }
    }
}