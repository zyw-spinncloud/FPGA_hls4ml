#include <iostream>

#include "MLP.h"
#include "parameters.h"


void MLP(
    input_t x[768],
    result_t layer4_out[768]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,layer4_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<fc1_weight_t, 2359296>(w2, "w2.txt");
        nnet::load_weights_from_txt<fc1_bias_t, 3072>(b2, "b2.txt");
        nnet::load_weights_from_txt<fc2_weight_t, 2359296>(w4, "w4.txt");
        nnet::load_weights_from_txt<fc2_bias_t, 768>(b4, "b4.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    fc1_result_t layer2_out[3072];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[3072];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    nnet::dense<input_t, fc1_result_t, config2>(x, layer2_out, w2, b2); // fc1

    nnet::relu<fc1_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // relu

    nnet::dense<layer3_t, result_t, config4>(layer3_out, layer4_out, w4, b4); // fc2

}

