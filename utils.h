//
// Created by Jeffery on 2018/11/28.
//

#ifndef SIMPLECNN_ACTIVATORS_H
#define SIMPLECNN_ACTIVATORS_H

#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;

/**
 * activate function
 * including its prime
 *
 * Because it is simple,hence make it a inline function
 */
inline double Relu(double input) {
    if (input) return input;
    return 0;
}

inline double Relu_prime(double output) {
    if (output > 0) return 1;
    return 0;
}

inline double Sigmoid(double input) {
    return 1.0 / (1.0 + exp(-input));
}

inline double Sigmoid_prime(double output) {
    double tmp = Sigmoid(output);
    return tmp * (1 - tmp);
}


/**
 *
 * cost funtion
 */

VectorXd cost_prime(VectorXd output_activation, VectorXd y);
#endif //SIMPLECNN_ACTIVATORS_H
