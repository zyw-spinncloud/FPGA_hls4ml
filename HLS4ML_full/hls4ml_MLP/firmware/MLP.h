#ifndef MLP_H_
#define MLP_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void MLP(
    input_t x[768],
    result_t layer4_out[768]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
