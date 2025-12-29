#pragma once
#include "ap_fixed.h"

static const int SEQ = 1;
static const int D   = 16;
static const int VEC = SEQ * D;
static const int NUM_LAYER = 8;      // number of (MHA->FFN) reuses after emb

using data_t = ap_fixed<16,6>;
