#ifndef MHA_TRANSFORMER_H_
#define MHA_TRANSFORMER_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void MHA_transformer(
    input_t mha_input[1*16],
    result_t layer19_out[1*16]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
