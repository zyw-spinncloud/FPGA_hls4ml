#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
// #include "parameters.h"

// C-Flag should be aligned with relative position
// header file of each sub-kernel should all be included
// eg. -I /home/ziyuanwang/manu_pg1
#include "emb/firmware/nnet_utils/nnet_types.h"
#include "MLP_micro/firmware/nnet_utils/nnet_types.h"

// Reuse variable with new name
// These sizes must match what you saw in the emb parameters header
using emb_elem_t = ap_fixed<16, 6>;
using mlp_elem_t = ap_fixed<34, 14>;

using emb_input_t  = nnet::array<emb_elem_t, 2>;     // token into emb
using emb_result_t = nnet::array<emb_elem_t, 16>;    // token out of emb
using mlp_input_t  = nnet::array<emb_elem_t, 16>;    // token into MLP
using mlp_result_t = nnet::array<mlp_elem_t, 64>;    // token out of MLP

// Prototypes matching emb project
void emb(hls::stream<emb_input_t>  &embedding_input,
         hls::stream<emb_result_t> &layer2_out);

// Prototypes matching MLP_micro project
void MLP_micro(
    hls::stream<mlp_input_t> &x,
    hls::stream<mlp_result_t> &layer7_out
);


// Pack emb input token (2x 16-bit) => 32-bit word
using in_word_t  = ap_uint<32>;
// Pack MLP output token: 64 * 34 = 2176 bits
// Put each output token across 2176 bits. Round up to 2304 (9*256) or 2304/2048 etc.
// We'll use 2304 bits here for alignment simplicity.
using out_word_t = ap_uint<512>;
static const int OUT_WORDS_PER_TOKEN = 5;

template<int W, int I>
static ap_uint<W> fixed_to_bits(ap_fixed<W, I> v) {
#pragma HLS INLINE
    ap_uint<W> b = v.range(W-1, 0);
    return b;
}

template<int W, int I>
static ap_fixed<W, I> bits_to_fixed(ap_uint<W> b) {
#pragma HLS INLINE
    ap_fixed<W, I> v;
    v.range(W-1, 0) = b;
    return v;
}

// -------- DDR -> stream (build emb_input_t tokens) --------
static void read_from_ddr(const in_word_t *in,
                          hls::stream<emb_input_t> &s_emb_in,
                          int n_tokens)
{
#pragma HLS INLINE off
    for (int t = 0; t < n_tokens; ++t) {
#pragma HLS PIPELINE II=1
        in_word_t w = in[t];

        ap_uint<16> b0 = w.range(15, 0);
        ap_uint<16> b1 = w.range(31, 16);

        emb_input_t tok;
        tok[0] = bits_to_fixed<16,6>(b0);
        tok[1] = bits_to_fixed<16,6>(b1);

        s_emb_in.write(tok);
    }
}

// -------- emb_result_t -> mlp_input_t (often identical) --------
static void bridge_emb_to_mlp(hls::stream<emb_result_t> &s_emb_out,
                              hls::stream<mlp_input_t>  &s_mlp_in,
                              int n_tokens)
{
#pragma HLS INLINE off
    for (int t = 0; t < n_tokens; ++t) {
#pragma HLS PIPELINE II=1
        emb_result_t e = s_emb_out.read();
        mlp_input_t  x;
        // If identical element type/size, this is just a copy:
        for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
            x[i] = e[i];
        }
        s_mlp_in.write(x);
    }
}

// -------- stream -> HP (pack mlp_result_t tokens) --------
static void write_to_ddr(hls::stream<mlp_result_t> &s_mlp_out,
                         out_word_t *out,
                         int n_tokens)
{
#pragma HLS INLINE off
    for (int t = 0; t < n_tokens; ++t) {
#pragma HLS PIPELINE II=1
        mlp_result_t r = s_mlp_out.read();

        // pack into 5x512-bit words
        ap_uint<2176> packed = 0;
        for (int i = 0; i < 64; ++i) {
#pragma HLS UNROLL
            ap_uint<34> bi = fixed_to_bits<34,14>(r[i]);
            packed.range(i*34 + 33, i*34) = bi;
        }

        // store 5 beats
        for (int k = 0; k < OUT_WORDS_PER_TOKEN; ++k) {
#pragma HLS UNROLL
            ap_uint<512> w = 0;
            int lo = k * 512;
            int hi = lo + 511;
            // guard for the last chunk (only 2176 bits valid)
            if (lo < 2176) {
                int src_hi = (hi < 2175) ? hi : 2175;
                w.range(src_hi - lo, 0) = packed.range(src_hi, lo);
            }
            out[t * OUT_WORDS_PER_TOKEN + k] = w;
        }
    }
}

// --------------- System-level top ---------------
void system_top(const in_word_t  *in_buf,
                out_word_t       *out_buf,
                int n_in_tokens,
                int n_out_tokens)
{
    #pragma HLS INTERFACE m_axi     port=in_buf   offset=slave bundle=gmem0 depth=1024
    #pragma HLS INTERFACE m_axi     port=out_buf  offset=slave bundle=gmem0 depth=1024

    #pragma HLS INTERFACE s_axilite port=in_buf       bundle=control
    #pragma HLS INTERFACE s_axilite port=out_buf      bundle=control
    #pragma HLS INTERFACE s_axilite port=n_in_tokens  bundle=control
    #pragma HLS INTERFACE s_axilite port=n_out_tokens bundle=control
    #pragma HLS INTERFACE s_axilite port=return       bundle=control

    #pragma HLS DATAFLOW

    hls::stream<emb_input_t>  s_emb_in("s_emb_in");
    hls::stream<emb_result_t> s_emb_out("s_emb_out");
    hls::stream<mlp_input_t>  s_mlp_in("s_mlp_in");
    hls::stream<mlp_result_t> s_mlp_out("s_mlp_out");

    #pragma HLS STREAM variable=s_emb_in   depth=64
    #pragma HLS STREAM variable=s_emb_out  depth=64
    #pragma HLS STREAM variable=s_mlp_in   depth=64
    #pragma HLS STREAM variable=s_mlp_out  depth=64

    read_from_ddr(in_buf, s_emb_in, n_in_tokens);
    emb(s_emb_in, s_emb_out);

    // Usually n_out_tokens == number of emb output tokens == number of mlp input tokens
    bridge_emb_to_mlp(s_emb_out, s_mlp_in, n_out_tokens);

    MLP_micro(s_mlp_in, s_mlp_out);
    write_to_ddr(s_mlp_out, out_buf, n_out_tokens);
}