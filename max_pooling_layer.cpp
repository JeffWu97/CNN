//
// Created by Jeffery on 2018/11/28.
//

#include "max_pooling_layer.h"
#include "fully_connected_layer.h"

max_pooling_layer::max_pooling_layer() {
    for (int i = 0; i < 20; i++) {
        activations[i] = MatrixXd::Zero(12, 12);
        deltas[i] = MatrixXd::Zero(12, 12);
    }
}

max_pooling_layer::~max_pooling_layer() {}

void max_pooling_layer::max_pooling(conv_layer conv_layer1) {
    MatrixXd *input = conv_layer1.activations;

    for (int i = 0; i < 20; i++)
        for (int j = 0; j < 12; j++)
            for (int k = 0; k < 12; k++) {
                double output = input[i].block<2, 2>(j, k).maxCoeff();
                activations[i].coeffRef(j, k) = output;
            }
}

VectorXd max_pooling_layer::flatten() {
    MatrixXd tmp(12 * 20, 12);
    for (int i = 0; i < 20; i++)
        tmp.block(i * 12, 0, 12, 12) = activations[i];

    tmp.transposeInPlace();
    VectorXd flatten_layer(Map<VectorXd>(tmp.data(), tmp.size()));
    return flatten_layer;
}

void max_pooling_layer::bp(fully_connected_layer *fully_connected_layer1) {

    VectorXd deltas_last = fully_connected_layer1->deltas;
    MatrixXd weights_last = fully_connected_layer1->weights;

    for (int i = 0; i < 20; i++)
        for (int j = 0; j < 12; j++)
            for (int k = 0; k < 12; k++) {
                deltas[i](j, k) = weights_last.col(j * 12 + k).dot(deltas_last);
            }

}