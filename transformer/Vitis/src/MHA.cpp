#include <iostream>

#include "MHA/firmware/MHA_transformer.h"
#include "MHA/firmware/parameters.h"


void MHA_transformer(
    input_t mha_input[1*16],
    result_t layer19_out[1*16]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=mha_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=mha_input,layer19_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<ln_attn_scale_t, 16>(s2, "s2.txt");
        nnet::load_weights_from_txt<ln_attn_bias_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<mha_query_weight_t, 256>(w4, "w4.txt");
        nnet::load_weights_from_txt<mha_query_bias_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<mha_key_weight_t, 256>(w7, "w7.txt");
        nnet::load_weights_from_txt<mha_key_bias_t, 16>(b7, "b7.txt");
        nnet::load_weights_from_txt<mha_value_weight_t, 256>(w10, "w10.txt");
        nnet::load_weights_from_txt<mha_value_bias_t, 16>(b10, "b10.txt");
        nnet::load_weights_from_txt<mha_attention_output_weight_t, 256>(w18, "w18.txt");
        nnet::load_weights_from_txt<mha_attention_output_bias_t, 16>(b18, "b18.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    ln_attn_result_t layer2_out[1*16];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[1*16];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer4_t layer4_out[1*4*4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    layer5_t layer5_out[1*4*4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[1*16];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    layer7_t layer7_out[1*4*4];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    layer8_t layer8_out[1*4*4];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    layer9_t layer9_out[1*16];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    layer10_t layer10_out[1*4*4];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    layer11_t layer11_out[1*4*4];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0

    layer12_t layer12_out[4*1*1];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    layer13_t layer13_out[4*1*1];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    layer14_t layer14_out[4*1*1];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    layer15_t layer15_out[4*1*1];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0

    layer16_t layer16_out[1*4*4];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0

    layer17_t layer17_out[1*4*4];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0

    layer18_t layer18_out[1*16];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0

    nnet::layernormalize<input_t, ln_attn_result_t, config2>(mha_input, layer2_out, s2, b2); // ln_attn

    nnet::mha_query_iq<ln_attn_result_t, layer3_t>(layer2_out, layer3_out); // mha_query_iq

    nnet::einsum_dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // mha_query

    nnet::mha_query_oq<layer4_t, layer5_t>(layer4_out, layer5_out); // mha_query_oq

    nnet::mha_key_iq<ln_attn_result_t, layer6_t>(layer2_out, layer6_out); // mha_key_iq

    nnet::einsum_dense<layer6_t, layer7_t, config7>(layer6_out, layer7_out, w7, b7); // mha_key

    nnet::mha_key_oq<layer7_t, layer8_t>(layer7_out, layer8_out); // mha_key_oq

    nnet::mha_value_iq<ln_attn_result_t, layer9_t>(layer2_out, layer9_out); // mha_value_iq

    nnet::einsum_dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // mha_value

    nnet::mha_value_oq<layer10_t, layer11_t>(layer10_out, layer11_out); // mha_value_oq

    nnet::einsum<layer8_t, layer5_t, layer12_t, config12>(layer8_out, layer5_out, layer12_out); // mha_mha_QK

    nnet::mha_q_softmax_iq<layer12_t, layer13_t>(layer12_out, layer13_out); // mha_q_softmax_iq

    nnet::softmax_multidim<layer13_t, layer14_t, softmax_config14>(layer13_out, layer14_out); // mha_q_softmax

    nnet::mha_q_softmax_oq<layer14_t, layer15_t>(layer14_out, layer15_out); // mha_q_softmax_oq

    nnet::einsum<layer15_t, layer11_t, layer16_t, config16>(layer15_out, layer11_out, layer16_out); // mha_mha_aV

    nnet::mha_attention_output_iq<layer16_t, layer17_t>(layer16_out, layer17_out); // mha_attention_output_iq

    nnet::einsum_dense<layer17_t, layer18_t, config18>(layer17_out, layer18_out, w18, b18); // mha_attention_output

    nnet::add<input_t, layer18_t, result_t, config19>(mha_input, layer18_out, layer19_out); // res_attn

}

