#ifndef FFN_TRANSFORMER_H_
#define FFN_TRANSFORMER_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void FFN_transformer(
    input_t x[1*16],
    result_t layer6_out[1*16]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
