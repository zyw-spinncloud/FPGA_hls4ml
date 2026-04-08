#include <iostream>

#include "MLP_micro.h"
#include "parameters.h"


void MLP_micro(
    hls::stream<input_t> &x,
    hls::stream<result_t> &layer7_out,
    //fc1_weight_t* w6,
    //fc1_bias_t*   b6,
    //fc2_weight_t* w7,
    //fc2_bias_t*   b7
    const model_default_t* params
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=x,layer7_out

    //#pragma HLS INTERFACE m_axi port=w6 offset=slave bundle=gmem0 depth=1024
    //#pragma HLS INTERFACE m_axi port=b6 offset=slave bundle=gmem0 depth=64
    //#pragma HLS INTERFACE m_axi port=w7 offset=slave bundle=gmem1 depth=1024
    //#pragma HLS INTERFACE m_axi port=b7 offset=slave bundle=gmem1 depth=16

    //#pragma HLS INTERFACE s_axilite port=w6 bundle=control
    //#pragma HLS INTERFACE s_axilite port=b6 bundle=control
    //#pragma HLS INTERFACE s_axilite port=w7 bundle=control
    //#pragma HLS INTERFACE s_axilite port=b7 bundle=control

    #pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem depth=240

    #pragma HLS INTERFACE s_axilite port=params     bundle=control
    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    static const int W6_LEN = 128;
    static const int B6_LEN = 64;
    static const int W7_LEN = 32;
    static const int B7_LEN = 16;

    static const int W6_OFF = 0;
    static const int B6_OFF = W6_OFF + W6_LEN;
    static const int W7_OFF = B6_OFF + B6_LEN;
    static const int B7_OFF = W7_OFF + W7_LEN;

    fc1_weight_t w6_cache[W6_LEN]; 
    fc1_bias_t   b6_cache[B6_LEN];
    fc2_weight_t w7_cache[W7_LEN];
    fc2_bias_t   b7_cache[B7_LEN];

    //load params
    for (int i = 0; i < W6_LEN; i++) {
        #pragma HLS PIPELINE II=1
        w6_cache[i] = (fc1_weight_t)params[W6_OFF + i];
    }
    for (int i = 0; i < B6_LEN; i++) {
        #pragma HLS PIPELINE II=1
        b6_cache[i] = (fc1_bias_t)params[B6_OFF + i];
    }
    for (int i = 0; i < W7_LEN; i++) {
        #pragma HLS PIPELINE II=1
        w7_cache[i] = (fc2_weight_t)params[W7_OFF + i];
    }
    for (int i = 0; i < B7_LEN; i++) {
        #pragma HLS PIPELINE II=1
        b7_cache[i] = (fc2_bias_t)params[B7_OFF + i];
    }

    //#pragma HLS DATAFLOW

    //#pragma HLS bind_storage variable=w6_cache type=ram_1p impl=bram
    //#pragma HLS bind_storage variable=b6_cache type=ram_1p impl=bram
    //#pragma HLS bind_storage variable=w7_cache type=ram_1p impl=bram
    //#pragma HLS bind_storage variable=b7_cache type=ram_1p impl=bram

    // hls-fpga-machine-learning insert load weights
//#ifndef __SYNTHESIS__
//    static bool loaded_weights = false;
//    if (!loaded_weights) {
//        nnet::load_weights_from_txt<fc1_weight_t, 1024>(w6, "w6.txt");
//        nnet::load_weights_from_txt<fc1_bias_t, 64>(b6, "b6.txt");
//        nnet::load_weights_from_txt<fc2_weight_t, 1024>(w7, "w7.txt");
//        nnet::load_weights_from_txt<fc2_bias_t, 16>(b7, "b7.txt");
//        loaded_weights = true;    }
//#endif
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

    nnet::pointwise_conv_1d_cl<layer5_t, fc1_result_t, config8>(layer5_out, layer6_out, w6_cache, b6_cache); // fc1

    nnet::relu<fc1_result_t, layer3_t, relu_config3>(layer6_out, layer3_out); // relu

    nnet::pointwise_conv_1d_cl<layer3_t, result_t, config9>(layer3_out, layer7_out, w7_cache, b7_cache); // fc2

}

