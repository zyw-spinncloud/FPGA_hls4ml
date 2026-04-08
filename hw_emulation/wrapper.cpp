#include "wrapper.h"
#include <iostream>

void wrapper(
    hls::stream<axis_pkt_t> &s_axis_in,
    hls::stream<axis_pkt_t> &m_axis_out,
    const model_default_t *params
) {
    #pragma HLS INTERFACE axis port=s_axis_in
    #pragma HLS INTERFACE axis port=m_axis_out

    #pragma HLS INTERFACE m_axi     port=params  offset=slave bundle=gmem depth=240
    #pragma HLS INTERFACE s_axilite port=params               bundle=control
    #pragma HLS INTERFACE s_axilite port=return               bundle=control

    hls::stream<input_t>  x_stream("x_stream");
    hls::stream<result_t> y_stream("y_stream");
    #pragma HLS STREAM variable=x_stream depth=4
    #pragma HLS STREAM variable=y_stream depth=4

    // 1) Read one 256-bit input beat from AXIS and convert to input_t
    // input_t = nnet::array<ap_fixed<16,6>, 16> = 256 bits total
  
    axis_pkt_t in_pkt = s_axis_in.read();

    input_t in_word;
    for (int i = 0; i < 16; i++) {
        #pragma HLS UNROLL
        ap_uint<16> bits = in_pkt.data.range((i + 1) * 16 - 1, i * 16);
        in_word[i].range(15, 0) = bits;
    }

    x_stream.write(in_word);

    // 2) Run the network
    MLP_micro(x_stream, y_stream, params);


    // 3) Read the result
    // result_t = nnet::array<ap_fixed<34,14>, 64> = 2176 bits total
    result_t out_word = y_stream.read();

    ap_uint<2176> packed_out = 0;

    for (int i = 0; i < 64; i++) {
        #pragma HLS UNROLL
        packed_out.range((i + 1) * 34 - 1, i * 34) = out_word[i].range(33, 0);
    }


    // 4) Send output over AXIS as 9 beats of 256 bits
    //    2176 bits = 8 full beats (2048) + 1 partial beat (128 bits)
    for (int beat = 0; beat < 9; beat++) {
        #pragma HLS PIPELINE II=1

        axis_pkt_t out_pkt;

        if (beat < 8) {
            out_pkt.data = packed_out.range((beat + 1) * 256 - 1, beat * 256);
            out_pkt.keep = 0xFFFFFFFF;   // all 32 bytes valid
            out_pkt.strb = 0xFFFFFFFF;
            out_pkt.last = 0;
        } else {
            // last beat: only lower 128 bits are valid
            ap_uint<256> last_data = 0;
            last_data.range(127, 0) = packed_out.range(2175, 2048);

            out_pkt.data = last_data;
            out_pkt.keep = 0x0000FFFF;   // lower 16 bytes valid
            out_pkt.strb = 0x0000FFFF;
            out_pkt.last = 1;
        }

        m_axis_out.write(out_pkt);
    }
}