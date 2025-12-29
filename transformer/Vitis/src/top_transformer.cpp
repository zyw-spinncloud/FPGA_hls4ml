#include "ap_int.h"
// #include "hls_stream.h"   // only needed if you switch to streams
#include "ap_fixed.h"
#include "top_types.h"

// // -------------------- Unified top-level types --------------------
// static const int SEQ = 1;
// static const int D   = 16;
// static const int VEC = SEQ * D;      // 16
// static const int NUM_LAYER = 8;      // number of (MHA->FFN) reuses after emb

// using data_t = ap_fixed<16, 6>;

// -------------------- Forward declarations (NO generated headers here) --------------------
// IMPORTANT: These signatures must match the generated ones in shape.
// Using concrete ap_fixed types avoids input_t/result_t name collisions.
void emb(data_t tokens[SEQ], data_t layer2_out[VEC]);

void MHA_transformer(data_t mha_input[VEC], data_t layer19_out[VEC]);

void FFN_transformer(data_t x[VEC], data_t layer6_out[VEC]);

// -------------------- Helpers --------------------
static void copy_vec(data_t dst[VEC], const data_t src[VEC]) {
#pragma HLS INLINE
    for (int i = 0; i < VEC; i++) {
#pragma HLS UNROLL
        dst[i] = src[i];
    }
}

// -------------------- Top --------------------
void top_transformer(data_t tokens[SEQ], data_t out[VEC]) {
#pragma HLS INTERFACE ap_memory port=tokens // token =1 could use ap_none
#pragma HLS INTERFACE ap_memory port=out
#pragma HLS INTERFACE ap_ctrl_hs port=return

    data_t layer_in[VEC];
    data_t layer_out[VEC];
#pragma HLS ARRAY_PARTITION variable=layer_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=layer_out complete dim=1

    // 1) emb -> layer_in
    emb(tokens, layer_in);

    // 2) Reuse loop: (MHA -> FFN) repeated NUM_LAYER times
    // No PIPELINE pragma here to avoid HLS trying to overlap iterations / replicate resources.
    for (int i = 0; i < NUM_LAYER; i++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
        MHA_transformer(layer_in, layer_out);
        FFN_transformer(layer_out, layer_in);
    }

    // 3) last_layer_in -> out
    copy_vec(out, layer_in);
}