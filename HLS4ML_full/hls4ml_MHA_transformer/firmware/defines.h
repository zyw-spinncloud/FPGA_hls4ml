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
typedef ap_fixed<33,13> ln_attn_result_t;
typedef ap_fixed<16,6> ln_attn_scale_t;
typedef ap_fixed<16,6> ln_attn_bias_t;
typedef ap_ufixed<8,5,AP_RND_CONV,AP_SAT,0> ln_attn_table_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> mha_query_accum_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> mha_query_weight_t;
typedef ap_fixed<16,6> mha_query_bias_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<16,6> mha_key_accum_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> mha_key_weight_t;
typedef ap_fixed<16,6> mha_key_bias_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<16,6> mha_value_accum_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<16,6> mha_value_weight_t;
typedef ap_fixed<16,6> mha_value_bias_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<16,6> mha_mha_qk_accum_t;
typedef ap_fixed<16,6> layer12_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_ufixed<4,2,AP_RND_CONV,AP_SAT,0> mha_q_softmax_exp_table_t;
typedef ap_ufixed<4,2,AP_RND_CONV,AP_SAT,0> mha_q_softmax_inv_table_t;
typedef ap_fixed<7,5,AP_RND,AP_WRAP,0> mha_q_softmax_inv_inp_t;
typedef ap_fixed<7,5,AP_RND,AP_WRAP,0> mha_q_softmax_inp_norm_t;
typedef ap_fixed<16,6> layer14_t;
typedef ap_fixed<18,8> mha_q_softmax_table_t;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<16,6> mha_mha_av_accum_t;
typedef ap_fixed<16,6> layer16_t;
typedef ap_fixed<16,6> layer17_t;
typedef ap_fixed<16,6> mha_attention_output_accum_t;
typedef ap_fixed<16,6> layer18_t;
typedef ap_fixed<16,6> mha_attention_output_weight_t;
typedef ap_fixed<16,6> mha_attention_output_bias_t;
typedef ap_fixed<16,6> result_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
