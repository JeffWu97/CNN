#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <ctime>


#include "max_pooling_layer.h"
#include "fully_connected_layer.h"
#include "conv_layer.h"
#include "utils.h"
#include "session.h"


using namespace std;

/**
 * this convolution neural network structure is specific for MNIST
 * and contain one conv_layer and one max pooling_layer , followed by a fully_connected_layer
 */


int main() {
    /*
     *  Configuration
     *  and model structure setup
     *  train.csv have totally 42000 pics
     */
    int epoch = 100;
    int mini_batch = 200;
    double eta = 1; //learning rate
    string file = "../data/train.csv";
    int number_of_train = 20000;
    int number_of_test = 300;

    conv_layer conv_layer1 = conv_layer();
    max_pooling_layer max_pooling_layer1 = max_pooling_layer();
    fully_connected_layer fully_connected_layer1 = fully_connected_layer();


    //read data from csv and store in the memory
    session session1 = session();
    session1.read_csv(file, number_of_train, number_of_test);

    /*
     * start training
     */
    clock_t start, finish;

    for (int i = 0; i < epoch; i++) {
        start = clock();

        for (int j = 0; j < mini_batch; j++) {
            //feed forward
            conv_layer1.convolution(session1.get_train_data() / 255);
            max_pooling_layer1.max_pooling(conv_layer1);
            fully_connected_layer1.feedforward(max_pooling_layer1);

            //back propagation

            fully_connected_layer1.bp(session1.get_train_label());
            max_pooling_layer1.bp(&fully_connected_layer1);
            conv_layer1.bp(&max_pooling_layer1);
        }

        //update weights and biases using SGD
        //test
//        cout << conv_layer1.deltas[0] << endl << endl;

        fully_connected_layer1.update(eta, mini_batch);
        conv_layer1.update(eta, mini_batch);

        finish = clock();

        /*
         * test the model
         */

        int correct_time = 0;
        session1.index_of_test_data = 0;
        for (int tmp = 0; tmp < number_of_test; tmp++) {
            //feed forward
            conv_layer1.convolution(session1.get_test_data() / 255);
            max_pooling_layer1.max_pooling(conv_layer1);
            fully_connected_layer1.feedforward(max_pooling_layer1);


            VectorXd::Index predict, label;
            fully_connected_layer1.activations.maxCoeff(&predict);
            session1.get_test_label().maxCoeff(&label);
            if (predict == label) correct_time++;
        }
        cout << "epoch " << i + 1 << "/" << epoch << " accuracy: " << (correct_time * 1.0 / number_of_test) * 100.0
             << "% "
             << (double) (finish - start) / (double) CLOCKS_PER_SEC << " s/epoch" << endl;

    }
}