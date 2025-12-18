#include <iostream>
// important to include the specific dir to avoid collision
#include "MLP_micro/firmware/MLP_micro.h"
#include "MLP_micro/firmware/parameters.h"
#include "MLP_micro/firmware/nnet_utils/nnet_types.h"



void MLP_micro(
    hls::stream<input_t> &x,
    hls::stream<result_t> &layer7_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=x,layer7_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<fc1_weight_t, 1024>(w6, "w6.txt");
        nnet::load_weights_from_txt<fc1_bias_t, 64>(b6, "b6.txt");
        nnet::load_weights_from_txt<fc2_weight_t, 1024>(w7, "w7.txt");
        nnet::load_weights_from_txt<fc2_bias_t, 16>(b7, "b7.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=16

    hls::stream<fc1_result_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=64

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=64

    nnet::transpose<input_t, layer5_t, config5>(x, layer5_out); // transpose_input_for_x

    nnet::pointwise_conv_1d_cl<layer5_t, fc1_result_t, config8>(layer5_out, layer6_out, w6, b6); // fc1

    nnet::relu<fc1_result_t, layer3_t, relu_config3>(layer6_out, layer3_out); // relu

    nnet::pointwise_conv_1d_cl<layer3_t, result_t, config9>(layer3_out, layer7_out, w7, b7); // fc2

}

