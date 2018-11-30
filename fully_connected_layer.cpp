//
// Created by Jeffery on 2018/11/28.
//

#include "fully_connected_layer.h"

fully_connected_layer::fully_connected_layer() {
    weights = MatrixXd::Random(10, 20 * 12 * 12);
    bias = VectorXd::Random(10);

    zs = VectorXd::Zero(10);
    activations = VectorXd::Zero(10);
    deltas = VectorXd::Zero(10);

    update_weights = MatrixXd::Zero(10, 20 * 12 * 12);
    update_bias = VectorXd::Zero(10);
}

void fully_connected_layer::feedforward(max_pooling_layer max_pooling_layer1) {

    activations_last = max_pooling_layer1.flatten();
    zs = weights * activations_last + bias;
    for (int i = 0; i < 10; i++)
        activations(i) = Sigmoid(zs(i));
}


void fully_connected_layer::bp(VectorXd y) {
    deltas = cost_prime(activations, y);

    for (int i = 0; i < 10; i++)
        deltas(i) *= Sigmoid_prime(zs(i));

    update_bias += deltas;

    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 20 * 12 * 12; j++)
            update_weights(i, j) += activations_last(j) * deltas(i);


}

void fully_connected_layer::update(double eta, int mini_batch) {
    bias -= eta / mini_batch * update_bias;
    weights -= eta / mini_batch * update_weights;

    //recover
    update_weights *= 0;
    update_bias *= 0;
}


