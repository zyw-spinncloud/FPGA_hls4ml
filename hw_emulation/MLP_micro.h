#ifndef MLP_MICRO_H_
#define MLP_MICRO_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void MLP_micro(
    hls::stream<input_t> &x,
    hls::stream<result_t> &layer7_out,
    //fc1_weight_t* w6,
    //fc1_bias_t*   b6,
    //fc2_weight_t* w7,
    //fc2_bias_t*   b7
    const model_default_t* params
);

// hls-fpga-machine-learning insert emulator-defines


#endif
