#ifndef MACHINE_LEARNING_LINALG_H
#define MACHINE_LEARNING_LINALG_H

#include <bits/stdc++.h>

std::ostream &operator<<(std::ostream &o, const std::vector<double> &x);

std::ostream &operator<<(std::ostream &o, const std::vector<std::vector<double>> &mat);

template <class T>
std::vector<T> operator*=(std::vector<T> &a, const std::vector<T> &b) {
    for (int i = 0; i < a.size(); ++i) a[i] *= b[i];
    return a;
}

template <class T>
std::vector<T> operator*(std::vector<T> a, const std::vector<T> &b) {
    return a *= b;
}

template <class T, class U>
std::vector<T> operator*=(std::vector<T> &a, const U &b) {
    for (auto &ai: a) ai *= b;
    return a;
}

template <class T, class U>
std::vector<T> operator*(std::vector<T> a, const U &b) {
    return a *= b;
}

template <class T>
std::vector<T> &operator+=(std::vector<T> &a, const std::vector<T> &b) {
    for (int i = 0; i < a.size(); ++i) a[i] += b[i];
    return a;
}

template <class T>
std::vector<T> operator+(std::vector<T> a, const std::vector<T> &b) {
    return a += b;
}

template <class T>
std::vector<T> &operator+=(std::vector<T> &a, const T &b) {
    for (auto &ai: a) ai += b;
    return a;
}

template <class T>
std::vector<T> operator+(std::vector<T> a, const T &b) {
    return a += b;
}

template <class T>
std::vector<T> &operator-=(std::vector<T> &a, const std::vector<T> &b) {
    for (int i = 0; i < a.size(); ++i) a[i] -= b[i];
    return a;
}

template <class T>
std::vector<T> operator-(std::vector<T> a, const std::vector<T> &b) {
    return a -= b;
}

template <class T>
std::vector<T> &operator-=(std::vector<T> &a, const T &b) {
    for (auto &ai: a) ai -= b;
    return a;
}

template <class T>
std::vector<T> operator-(std::vector<T> a, const T &b) {
    return a -= b;
}

template <class T>
std::vector<T> operator-(const T &a, std::vector<T> b) {
    for (auto &bi: b) bi = a - bi;
    return b;
}

template <class T, class U>
std::vector<T> operator/=(std::vector<T> &a, const U &b) {
    for (auto &ai: a) ai /= b;
    return a;
}

template <class T, class U>
std::vector<T> operator/(std::vector<T> a, const U &b) {
    return a /= b;
}

template <class T>
std::vector<T> operator-(std::vector<T> a) {
    for (auto &ai: a) ai = -ai;
    return a;
}

double dot(const std::vector<double> &a, const std::vector<double> &b);

template <class T>
T dot(const std::vector<T> &a, const std::vector<T> &b) {
    T ret = a[0] * b[0];
    for (int i = 0; ++i < a.size();) ret += a[i] * b[i];
    return ret;
}

template <class T>
T sum(const std::vector<T> &a) {
    return accumulate(++a.begin(), a.end(), *a.begin());
}

template <class T>
std::vector<T> exp(const std::vector<T> &x) {
    std::vector<T> ret;
    for (const auto &xi: x) {
        ret.push_back(exp(xi));
    }
    return ret;
}

#endif //MACHINE_LEARNING_LINALG_H
