#include <iostream>
// important to include the specific dir to avoid collision
#include "emb/firmware/emb.h"
#include "emb/firmware/parameters.h"
#include "emb/firmware/nnet_utils/nnet_types.h"




void emb(
    hls::stream<input_t> &embedding_input,
    hls::stream<result_t> &layer2_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=embedding_input,layer2_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<model_default_t, 512>(e2, "e2.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    nnet::embedding<input_t, result_t, config2>(embedding_input, layer2_out, e2); // embedding

}

