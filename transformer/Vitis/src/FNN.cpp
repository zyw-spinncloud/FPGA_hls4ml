#include <iostream>

#include "FFN/firmware/FFN_transformer.h"
#include "FFN/firmware/parameters.h"


void FFN_transformer(
    input_t x[1*16],
    result_t layer6_out[1*16]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,layer6_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<ln_scale_t, 16>(s2, "s2.txt");
        nnet::load_weights_from_txt<ln_bias_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<fc1_weight_t, 1024>(w7, "w7.txt");
        nnet::load_weights_from_txt<fc1_bias_t, 64>(b7, "b7.txt");
        nnet::load_weights_from_txt<fc2_weight_t, 1024>(w8, "w8.txt");
        nnet::load_weights_from_txt<fc2_bias_t, 16>(b8, "b8.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    ln_result_t layer2_out[1*16];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    fc1_result_t layer7_out[1*64];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    layer4_t layer4_out[1*64];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    fc2_result_t layer8_out[1*16];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    nnet::layernormalize<input_t, ln_result_t, config2>(x, layer2_out, s2, b2); // ln

    nnet::pointwise_conv_1d_cl<ln_result_t, fc1_result_t, config9>(layer2_out, layer7_out, w7, b7); // fc1

    nnet::relu<fc1_result_t, layer4_t, relu_config4>(layer7_out, layer4_out); // relu

    nnet::pointwise_conv_1d_cl<layer4_t, fc2_result_t, config10>(layer4_out, layer8_out, w8, b8); // fc2

    nnet::add<input_t, fc2_result_t, result_t, config6>(x, layer8_out, layer6_out); // add

}

