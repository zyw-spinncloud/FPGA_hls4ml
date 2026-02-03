#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_embed.h"
#include "nnet_utils/nnet_embed_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/e2.h"


// hls-fpga-machine-learning insert layer-config
// tok_emb
struct config2 : nnet::embed_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 16;
    static const unsigned vocab_size = 32;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
    typedef model_default_t embeddings_t;
};



#endif
