// Top module to connect two sub component together.
// Here "top" use axi (,/s) interface for host communication.
// Submodule "emb" and "MLP_micro" communicate through axis.
// Notes:
// A. Header file of sub-component is not needed.
// B. C-Flag point to workplace eg. -I /home/ziyuanwang/manu_pg1
// C. When declaring the interface the variable
//    TYPE and WIDTH (EVERYTHING) should align with the definition of sub component.
//      Check define.h of each sub component to validate, eg.:
//          typedef nnet::array<ap_fixed<16,6>, 16*1> input_t;
//          typedef nnet::array<ap_fixed<34,14>, 64*1> result_t;
//                                       ^^^^^   ^^^^
//                  ^^^^^^^^^^^^^^^^^^^^
// D. sub component source file (cpp) must be added

#include <hls_stream.h>
#include <ap_int.h>
#include "ap_fixed.h"
// #include "parameters.h"

#include "emb/firmware/nnet_utils/nnet_types.h"

// Reuse variable with new name
// These sizes must match what you saw in the emb parameters header
using emb_input_t  = nnet::array<ap_fixed<16,6>, 2*1>;
using emb_result_t = nnet::array<ap_fixed<16,6>, 16*1>;

// These must match what you saw in the MLP_micro parameters header
using mlp_input_t  = nnet::array<ap_fixed<16,6>, 16*1>;
using mlp_result_t = nnet::array<ap_fixed<34,14>, 64*1>;

// Prototypes matching emb project
void emb(hls::stream<emb_input_t>  &embedding_input,
         hls::stream<emb_result_t> &layer2_out);

// Prototypes matching MLP_micro project
void MLP_micro(
    hls::stream<mlp_input_t> &x,
    hls::stream<mlp_result_t> &layer7_out
);

#define N_IN   13  // number of input elements (sequence length)
#define N_OUT  16  // number of output elements (output length)

// Help function

void read_input(emb_input_t in_buf[N_IN],
                hls::stream<emb_input_t> &s_emb_in) {
#pragma HLS INLINE off
    for (int i = 0; i < N_IN; ++i) {
    #pragma HLS PIPELINE II=1
        s_emb_in.write(in_buf[i]);
    }
}

// NOTE: double check *right* loop length here:
void convert_emb_to_mlp(hls::stream<emb_result_t> &s_emb_out,
                        hls::stream<mlp_input_t>  &s_mlp_in) {
#pragma HLS INLINE off
    for (int i = 0; i < N_OUT; ++i) {  // OR another constant that matches emb output length
    #pragma HLS PIPELINE II=1
        
        emb_result_t e = s_emb_out.read();
        mlp_input_t  x;
        //mlp_input_t  x = (mlp_input_t)e;   // cast if types differ; if identical type, no cast needed
        x[0] = e[0];
        x[1] = e[1];
        
        s_mlp_in.write(x);
    }
}

// NOTE: double check *right* loop length here:
void write_output(hls::stream<mlp_result_t> &s_mlp_out,
                  mlp_result_t out_buf[N_OUT]) {
#pragma HLS INLINE off

    for (int i = 0; i < N_OUT; ++i) {
    #pragma HLS PIPELINE II=1
        mlp_result_t tmp = s_mlp_out.read();
        for (int j = 0; j < 64; ++j) {  // replace 64 with MLP_OUT_SIZE
        #pragma HLS UNROLL
            out_buf[i][j] = tmp[j];
        }
    }
}


// --------------- System-level top ---------------
void system_top(emb_input_t  in_buf[N_IN],
                    mlp_result_t out_buf[N_OUT]) {
    // === 1. CPU-visible non-stream I/O (DDR/AXI) ===
    #pragma HLS INTERFACE m_axi     port=in_buf  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi     port=out_buf offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=in_buf  bundle=control
    #pragma HLS INTERFACE s_axilite port=out_buf bundle=control
    #pragma HLS INTERFACE s_axilite port=return  bundle=control

    // === 2. Internal dataflow / streaming ===
    #pragma HLS DATAFLOW

    hls::stream<emb_input_t>  s_emb_in("s_emb_in");
    hls::stream<emb_result_t> s_emb_out("s_emb_out");
    hls::stream<mlp_input_t>  s_mlp_in("s_mlp_in");
    hls::stream<mlp_result_t> s_mlp_out("s_mlp_out");

    #pragma HLS STREAM variable=s_emb_in   depth=32
    #pragma HLS STREAM variable=s_emb_out  depth=32
    #pragma HLS STREAM variable=s_mlp_in   depth=32
    #pragma HLS STREAM variable=s_mlp_out  depth=32

    // 1) DDR -> stream for emb
    read_input(in_buf, s_emb_in);

    // 2) emb block (stream in -> stream out)
    emb(s_emb_in, s_emb_out);

    // 3) convert emb output to MLP input (if needed)
    convert_emb_to_mlp(s_emb_out, s_mlp_in);

    // 4) MLP block
    MLP_micro(s_mlp_in, s_mlp_out);

    // 5) stream -> DDR for host
    write_output(s_mlp_out, out_buf);
}