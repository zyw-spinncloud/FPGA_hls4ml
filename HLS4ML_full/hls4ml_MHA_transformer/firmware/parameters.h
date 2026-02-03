#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_einsum.h"
#include "nnet_utils/nnet_einsum_dense.h"
#include "nnet_utils/nnet_layernorm.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/s2.h"
#include "weights/b2.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w7.h"
#include "weights/b7.h"
#include "weights/w10.h"
#include "weights/b10.h"
#include "weights/w18.h"
#include "weights/b18.h"


// hls-fpga-machine-learning insert layer-config
// ln_attn
struct config2 : nnet::layernorm_config {
    static const unsigned n_in = 1*16;
    static const unsigned seq_len = 1;
    static const unsigned axis = 2;
    static const unsigned epsilon_power_of_10 = 5;
    static const unsigned table_range_power2 = 0;
    static const unsigned table_size = 4096;
    typedef model_default_t accum_t;
    typedef ln_attn_bias_t bias_t;
    typedef ln_attn_scale_t scale_t;
    typedef ln_attn_table_t table_t;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// mha_query
struct config4_tpose_inp {
    static const unsigned dims = 2;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config4_tpose_inp_from_shape[2] = {1, 16};
unsigned config4_tpose_inp_to_shape[2] = {1, 16};
unsigned config4_tpose_inp_perm[2] = {0, 1};
unsigned config4_tpose_inp_perm_strides[2] = {16, 1};

const unsigned* const config4_tpose_inp::from_shape = config4_tpose_inp_from_shape;
const unsigned* const config4_tpose_inp::to_shape = config4_tpose_inp_to_shape;
const unsigned* const config4_tpose_inp::perm = config4_tpose_inp_perm;
const unsigned* const config4_tpose_inp::perm_strides = config4_tpose_inp_perm_strides;


struct config4_tpose_out {
    static const unsigned dims = 3;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config4_tpose_out_from_shape[3] = {1, 4, 4};
unsigned config4_tpose_out_to_shape[3] = {1, 4, 4};
unsigned config4_tpose_out_perm[3] = {0, 1, 2};
unsigned config4_tpose_out_perm_strides[3] = {16, 4, 1};

const unsigned* const config4_tpose_out::from_shape = config4_tpose_out_from_shape;
const unsigned* const config4_tpose_out::to_shape = config4_tpose_out_to_shape;
const unsigned* const config4_tpose_out::perm = config4_tpose_out_perm;
const unsigned* const config4_tpose_out::perm_strides = config4_tpose_out_perm_strides;


struct config4_dense : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 16;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 133;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef mha_query_accum_t accum_t;
    typedef mha_query_bias_t bias_t;
    typedef mha_query_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config4 {
    typedef config4_tpose_inp tpose_inp_conf;
    typedef config4_tpose_out tpose_out_conf;

    typedef mha_query_accum_t accum_t;
    typedef mha_query_bias_t bias_t;

    typedef config4_dense dense_conf;

    // Layer Sizes
    static const unsigned n_free_data = 1;
    static const unsigned n_free_kernel = 16;
    static const unsigned n_contract = 16;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 16;
    static const unsigned parallelization_factor = 1; // Only useful when n_inplace > 1
};

// mha_key
struct config7_tpose_inp {
    static const unsigned dims = 2;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config7_tpose_inp_from_shape[2] = {1, 16};
unsigned config7_tpose_inp_to_shape[2] = {1, 16};
unsigned config7_tpose_inp_perm[2] = {0, 1};
unsigned config7_tpose_inp_perm_strides[2] = {16, 1};

const unsigned* const config7_tpose_inp::from_shape = config7_tpose_inp_from_shape;
const unsigned* const config7_tpose_inp::to_shape = config7_tpose_inp_to_shape;
const unsigned* const config7_tpose_inp::perm = config7_tpose_inp_perm;
const unsigned* const config7_tpose_inp::perm_strides = config7_tpose_inp_perm_strides;


struct config7_tpose_out {
    static const unsigned dims = 3;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config7_tpose_out_from_shape[3] = {1, 4, 4};
unsigned config7_tpose_out_to_shape[3] = {1, 4, 4};
unsigned config7_tpose_out_perm[3] = {0, 1, 2};
unsigned config7_tpose_out_perm_strides[3] = {16, 4, 1};

const unsigned* const config7_tpose_out::from_shape = config7_tpose_out_from_shape;
const unsigned* const config7_tpose_out::to_shape = config7_tpose_out_to_shape;
const unsigned* const config7_tpose_out::perm = config7_tpose_out_perm;
const unsigned* const config7_tpose_out::perm_strides = config7_tpose_out_perm_strides;


struct config7_dense : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 16;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 150;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef mha_key_accum_t accum_t;
    typedef mha_key_bias_t bias_t;
    typedef mha_key_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config7 {
    typedef config7_tpose_inp tpose_inp_conf;
    typedef config7_tpose_out tpose_out_conf;

    typedef mha_key_accum_t accum_t;
    typedef mha_key_bias_t bias_t;

    typedef config7_dense dense_conf;

    // Layer Sizes
    static const unsigned n_free_data = 1;
    static const unsigned n_free_kernel = 16;
    static const unsigned n_contract = 16;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 16;
    static const unsigned parallelization_factor = 1; // Only useful when n_inplace > 1
};

// mha_value
struct config10_tpose_inp {
    static const unsigned dims = 2;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config10_tpose_inp_from_shape[2] = {1, 16};
unsigned config10_tpose_inp_to_shape[2] = {1, 16};
unsigned config10_tpose_inp_perm[2] = {0, 1};
unsigned config10_tpose_inp_perm_strides[2] = {16, 1};

const unsigned* const config10_tpose_inp::from_shape = config10_tpose_inp_from_shape;
const unsigned* const config10_tpose_inp::to_shape = config10_tpose_inp_to_shape;
const unsigned* const config10_tpose_inp::perm = config10_tpose_inp_perm;
const unsigned* const config10_tpose_inp::perm_strides = config10_tpose_inp_perm_strides;


struct config10_tpose_out {
    static const unsigned dims = 3;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config10_tpose_out_from_shape[3] = {1, 4, 4};
unsigned config10_tpose_out_to_shape[3] = {1, 4, 4};
unsigned config10_tpose_out_perm[3] = {0, 1, 2};
unsigned config10_tpose_out_perm_strides[3] = {16, 4, 1};

const unsigned* const config10_tpose_out::from_shape = config10_tpose_out_from_shape;
const unsigned* const config10_tpose_out::to_shape = config10_tpose_out_to_shape;
const unsigned* const config10_tpose_out::perm = config10_tpose_out_perm;
const unsigned* const config10_tpose_out::perm_strides = config10_tpose_out_perm_strides;


struct config10_dense : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 16;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 151;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef mha_value_accum_t accum_t;
    typedef mha_value_bias_t bias_t;
    typedef mha_value_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config10 {
    typedef config10_tpose_inp tpose_inp_conf;
    typedef config10_tpose_out tpose_out_conf;

    typedef mha_value_accum_t accum_t;
    typedef mha_value_bias_t bias_t;

    typedef config10_dense dense_conf;

    // Layer Sizes
    static const unsigned n_free_data = 1;
    static const unsigned n_free_kernel = 16;
    static const unsigned n_contract = 16;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 16;
    static const unsigned parallelization_factor = 1; // Only useful when n_inplace > 1
};

// mha_mha_QK
struct config12_tpose_inp0 {
    static const unsigned dims = 3;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config12_tpose_inp0_from_shape[3] = {1, 4, 4};
unsigned config12_tpose_inp0_to_shape[3] = {4, 1, 4};
unsigned config12_tpose_inp0_perm[3] = {1, 0, 2};
unsigned config12_tpose_inp0_perm_strides[3] = {4, 16, 1};

const unsigned* const config12_tpose_inp0::from_shape = config12_tpose_inp0_from_shape;
const unsigned* const config12_tpose_inp0::to_shape = config12_tpose_inp0_to_shape;
const unsigned* const config12_tpose_inp0::perm = config12_tpose_inp0_perm;
const unsigned* const config12_tpose_inp0::perm_strides = config12_tpose_inp0_perm_strides;


struct config12_tpose_inp1 {
    static const unsigned dims = 3;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config12_tpose_inp1_from_shape[3] = {1, 4, 4};
unsigned config12_tpose_inp1_to_shape[3] = {4, 1, 4};
unsigned config12_tpose_inp1_perm[3] = {1, 0, 2};
unsigned config12_tpose_inp1_perm_strides[3] = {4, 16, 1};

const unsigned* const config12_tpose_inp1::from_shape = config12_tpose_inp1_from_shape;
const unsigned* const config12_tpose_inp1::to_shape = config12_tpose_inp1_to_shape;
const unsigned* const config12_tpose_inp1::perm = config12_tpose_inp1_perm;
const unsigned* const config12_tpose_inp1::perm_strides = config12_tpose_inp1_perm_strides;


struct config12_tpose_out {
    static const unsigned dims = 3;
    static const unsigned N = 4;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config12_tpose_out_from_shape[3] = {4, 1, 1};
unsigned config12_tpose_out_to_shape[3] = {4, 1, 1};
unsigned config12_tpose_out_perm[3] = {0, 2, 1};
unsigned config12_tpose_out_perm_strides[3] = {1, 1, 1};

const unsigned* const config12_tpose_out::from_shape = config12_tpose_out_from_shape;
const unsigned* const config12_tpose_out::to_shape = config12_tpose_out_to_shape;
const unsigned* const config12_tpose_out::perm = config12_tpose_out_perm;
const unsigned* const config12_tpose_out::perm_strides = config12_tpose_out_perm_strides;



struct config12 {
    typedef config12_tpose_inp0 tpose_inp0_config;
    typedef config12_tpose_inp1 tpose_inp1_config;
    typedef config12_tpose_out tpose_out_conf;

    typedef mha_mha_qk_accum_t accum_t;

    // Layer Sizes
    static const unsigned n_free0 = 1;
    static const unsigned n_free1 = 1;
    static const unsigned n_contract = 4;
    static const unsigned n_inplace = 4;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 16;
    static const unsigned multiplier_limit = 1;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// mha_q_softmax
struct softmax_config14 : nnet::activ_config {
    static const unsigned n_in = 4;
    static const unsigned n_slice = 1;
    static const unsigned n_outer = 4;
    static const unsigned n_inner = 1;
    static const unsigned parallelization_factor = 4;
    static const unsigned exp_table_size = 64;
    static const unsigned inv_table_size = 128;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    static constexpr float exp_scale = 0.5;
    typedef mha_q_softmax_exp_table_t exp_table_t;
    typedef mha_q_softmax_inv_table_t inv_table_t;
    typedef model_default_t accum_t;
    typedef mha_q_softmax_inv_inp_t inv_inp_t;
    typedef mha_q_softmax_inp_norm_t inp_norm_t;
};

// mha_mha_aV
struct config16_tpose_inp0 {
    static const unsigned dims = 3;
    static const unsigned N = 4;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config16_tpose_inp0_from_shape[3] = {4, 1, 1};
unsigned config16_tpose_inp0_to_shape[3] = {4, 1, 1};
unsigned config16_tpose_inp0_perm[3] = {0, 1, 2};
unsigned config16_tpose_inp0_perm_strides[3] = {1, 1, 1};

const unsigned* const config16_tpose_inp0::from_shape = config16_tpose_inp0_from_shape;
const unsigned* const config16_tpose_inp0::to_shape = config16_tpose_inp0_to_shape;
const unsigned* const config16_tpose_inp0::perm = config16_tpose_inp0_perm;
const unsigned* const config16_tpose_inp0::perm_strides = config16_tpose_inp0_perm_strides;


struct config16_tpose_inp1 {
    static const unsigned dims = 3;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config16_tpose_inp1_from_shape[3] = {1, 4, 4};
unsigned config16_tpose_inp1_to_shape[3] = {4, 4, 1};
unsigned config16_tpose_inp1_perm[3] = {1, 2, 0};
unsigned config16_tpose_inp1_perm_strides[3] = {4, 1, 16};

const unsigned* const config16_tpose_inp1::from_shape = config16_tpose_inp1_from_shape;
const unsigned* const config16_tpose_inp1::to_shape = config16_tpose_inp1_to_shape;
const unsigned* const config16_tpose_inp1::perm = config16_tpose_inp1_perm;
const unsigned* const config16_tpose_inp1::perm_strides = config16_tpose_inp1_perm_strides;


struct config16_tpose_out {
    static const unsigned dims = 3;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config16_tpose_out_from_shape[3] = {4, 1, 4};
unsigned config16_tpose_out_to_shape[3] = {1, 4, 4};
unsigned config16_tpose_out_perm[3] = {1, 0, 2};
unsigned config16_tpose_out_perm_strides[3] = {4, 4, 1};

const unsigned* const config16_tpose_out::from_shape = config16_tpose_out_from_shape;
const unsigned* const config16_tpose_out::to_shape = config16_tpose_out_to_shape;
const unsigned* const config16_tpose_out::perm = config16_tpose_out_perm;
const unsigned* const config16_tpose_out::perm_strides = config16_tpose_out_perm_strides;



struct config16 {
    typedef config16_tpose_inp0 tpose_inp0_config;
    typedef config16_tpose_inp1 tpose_inp1_config;
    typedef config16_tpose_out tpose_out_conf;

    typedef mha_mha_av_accum_t accum_t;

    // Layer Sizes
    static const unsigned n_free0 = 1;
    static const unsigned n_free1 = 4;
    static const unsigned n_contract = 1;
    static const unsigned n_inplace = 4;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 16;
    static const unsigned multiplier_limit = 1;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// mha_attention_output
struct config18_tpose_inp {
    static const unsigned dims = 3;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config18_tpose_inp_from_shape[3] = {1, 4, 4};
unsigned config18_tpose_inp_to_shape[3] = {1, 4, 4};
unsigned config18_tpose_inp_perm[3] = {0, 1, 2};
unsigned config18_tpose_inp_perm_strides[3] = {16, 4, 1};

const unsigned* const config18_tpose_inp::from_shape = config18_tpose_inp_from_shape;
const unsigned* const config18_tpose_inp::to_shape = config18_tpose_inp_to_shape;
const unsigned* const config18_tpose_inp::perm = config18_tpose_inp_perm;
const unsigned* const config18_tpose_inp::perm_strides = config18_tpose_inp_perm_strides;


struct config18_tpose_out {
    static const unsigned dims = 2;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config18_tpose_out_from_shape[2] = {1, 16};
unsigned config18_tpose_out_to_shape[2] = {1, 16};
unsigned config18_tpose_out_perm[2] = {0, 1};
unsigned config18_tpose_out_perm_strides[2] = {16, 1};

const unsigned* const config18_tpose_out::from_shape = config18_tpose_out_from_shape;
const unsigned* const config18_tpose_out::to_shape = config18_tpose_out_to_shape;
const unsigned* const config18_tpose_out::perm = config18_tpose_out_perm;
const unsigned* const config18_tpose_out::perm_strides = config18_tpose_out_perm_strides;


struct config18_dense : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 16;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 111;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef mha_attention_output_accum_t accum_t;
    typedef mha_attention_output_bias_t bias_t;
    typedef mha_attention_output_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config18 {
    typedef config18_tpose_inp tpose_inp_conf;
    typedef config18_tpose_out tpose_out_conf;

    typedef mha_attention_output_accum_t accum_t;
    typedef mha_attention_output_bias_t bias_t;

    typedef config18_dense dense_conf;

    // Layer Sizes
    static const unsigned n_free_data = 1;
    static const unsigned n_free_kernel = 16;
    static const unsigned n_contract = 16;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 16;
    static const unsigned parallelization_factor = 1; // Only useful when n_inplace > 1
};

// res_attn
struct config19 : nnet::merge_config {
    static const unsigned n_elem = 1*16;
    static const unsigned reuse_factor = 16;
};



#endif
