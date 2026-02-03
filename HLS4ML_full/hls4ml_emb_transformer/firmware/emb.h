#ifndef EMB_H_
#define EMB_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void emb(
    input_t tokens[1],
    result_t layer2_out[1*16]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
