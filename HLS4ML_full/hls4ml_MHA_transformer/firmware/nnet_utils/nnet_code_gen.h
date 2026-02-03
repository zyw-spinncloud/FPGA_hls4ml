#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include "nnet_conv1d_latency.h"
#include "nnet_helpers.h"

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_function_stubs.h"
#include "nnet_mult.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T> class PointwiseConv1D {
  public:
    static void pointwise_conv(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                               res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        // To be implemented in subclasses
    }
};

// hls4ml insert code

template<typename input_t, typename output_t>
void mha_query_iq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[5]);
    out[6] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[6]);
    out[7] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[9]);
    out[10] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[10]);
    out[11] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[15]);
}

template<typename input_t, typename output_t>
void mha_query_oq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[5]);
    out[6] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[6]);
    out[7] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[9]);
    out[10] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[10]);
    out[11] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[15]);
}

template<typename input_t, typename output_t>
void mha_key_iq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[5]);
    out[6] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[6]);
    out[7] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[9]);
    out[10] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[10]);
    out[11] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[15]);
}

template<typename input_t, typename output_t>
void mha_key_oq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[5]);
    out[6] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[6]);
    out[7] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[9]);
    out[10] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[10]);
    out[11] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[15]);
}

template<typename input_t, typename output_t>
void mha_value_iq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[5]);
    out[6] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[6]);
    out[7] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[9]);
    out[10] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[10]);
    out[11] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[15]);
}

template<typename input_t, typename output_t>
void mha_value_oq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[5]);
    out[6] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[6]);
    out[7] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[9]);
    out[10] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[10]);
    out[11] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[15]);
}

template<typename input_t, typename output_t>
void mha_q_softmax_iq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[3]);
}

template<typename input_t, typename output_t>
void mha_q_softmax_oq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[3]);
}

template<typename input_t, typename output_t>
void mha_attention_output_iq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[5]);
    out[6] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[6]);
    out[7] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[9]);
    out[10] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[10]);
    out[11] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[15]);
}

} // namespace nnet

#endif
