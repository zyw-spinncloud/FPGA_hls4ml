#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<33,13> ln_result_t;
typedef ap_fixed<16,6> ln_scale_t;
typedef ap_fixed<16,6> ln_bias_t;
typedef ap_ufixed<8,5,AP_RND_CONV,AP_SAT,0> ln_table_t;
typedef ap_fixed<54,24> fc1_result_t;
typedef ap_fixed<16,6> fc1_weight_t;
typedef ap_fixed<16,6> fc1_bias_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<18,8> relu_table_t;
typedef ap_fixed<39,19> fc2_result_t;
typedef ap_fixed<16,6> fc2_weight_t;
typedef ap_fixed<16,6> fc2_bias_t;
typedef ap_fixed<16,6> result_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
