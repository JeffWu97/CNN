//
// Created by Jeffery on 2018/11/28.
//

#include "utils.h"

VectorXd cost_prime(VectorXd output_activation, VectorXd y) {
    return output_activation - y;
}

