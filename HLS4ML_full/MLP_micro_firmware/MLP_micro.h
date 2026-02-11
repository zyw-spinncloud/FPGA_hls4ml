#ifndef MLP_MICRO_H_
#define MLP_MICRO_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void MLP_micro(
    hls::stream<input_t> &x,
    hls::stream<result_t> &layer7_out
);

// hls-fpga-machine-learning insert emulator-defines


#endif
