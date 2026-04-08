#ifndef WRAPPER_H_
#define WRAPPER_H_

#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"
#include "MLP_micro.h"

typedef ap_axiu<256, 0, 0, 0> axis_pkt_t;

void wrapper(
    hls::stream<axis_pkt_t> &s_axis_in,
    hls::stream<axis_pkt_t> &m_axis_out,
    const model_default_t *params
);

#endif