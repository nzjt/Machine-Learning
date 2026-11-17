#include "linalg.h"

std::ostream &operator<<(std::ostream &o, const std::vector<double> &x) {
    o << '[';
    for (auto &xi: x) o << xi << ' ';
    return o << "\b]";
}

std::ostream &operator<<(std::ostream &o, const std::vector<std::vector<double>> &mat) {
    o << '[';
    for (int i = 0; i < mat.size(); ++i) {
        for (int j = 0; j < mat[i].size(); ++j) {
            if (i or j) {
                o << ' ';
            }
            o << mat[i][j];
        }
        if (i + 1 == mat.size()) {
            o << ']';
        } else {
            o << std::endl;
        }
    }
    return o;
}