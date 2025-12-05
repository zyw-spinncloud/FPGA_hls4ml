#include <iostream>
#include <cmath>
#include <unistd.h>           // chdir, getcwd

#include "MLP.h"     // DUT
#include "mlp_golden_data.h"  // BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, x_embed_golden, y_out_golden
#include "defines.h" // input_t, result_t, model_default_t

int main() {
    // 1) Change cwd so MLP.cpp can find w6.txt, b6.txt, ...
    const char* fw_dir = "/home/ziyuanwang/vitis_hls_mlp/MLP/firmware";
    if (chdir(fw_dir) != 0) {
        perror("chdir to firmware dir failed");
        return 1;
    }

    // hls::stream<input_t>  x_stream;
    // hls::stream<result_t> y_stream;

    // // 2) Push SEQ_LEN tokens into the stream (BATCH_SIZE is assumed 1)
    // for (int t = 0; t < SEQ_LEN; ++t) {
    //     input_t word;
    //     for (int h = 0; h < HIDDEN_SIZE; ++h) {
    //         word[h] = (model_default_t)x_embed_golden[t][h];
    //     }
    //     x_stream.write(word);
    // }

    input_t in[768];
    result_t out[768];

    // Global stats over all tokens & hidden units
    double sum_abs_err = 0.0;
    int    total_elems = 0;

    // Loop over tokens (SEQ_LEN)
    for (int t = 0; t < SEQ_LEN; t++) {
        // Load one token (768 features) into the parallel input vector
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            in[h] = x_embed_golden[t][h];  // cast to fixed if needed
        }

        // 3) Call DUT
        MLP(in, out);

        // Compare result with golden
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            float hw = (float) out[h]; // convert from ap_fixed
            float ref = y_out_golden[t][h];

            float diff = std::fabs (hw - ref);

            sum_abs_err += diff;
            ++total_elems;
        }

    }

    double mae  = sum_abs_err / total_elems;   // Mean Absolute Error

    std::cout << "Total elements  = " << total_elems << "\n";
    std::cout << "MAE             = " << mae          << "\n";

    // common MAE vs float model in precision ap_fixed<16,6>
    // calculated following quantization and accumulation of MLP:
    // see documentation for details, 0.02 - 0.03 is a proper range
    if (mae > 0.027) {
        std::cout << "TEST FAILED\n";
        return 1;
    } else {
        std::cout << "TEST PASSED\n";
        return 0;
    }

    // // if using stream 
    // while (!y_stream.empty()) {
    //     result_t y_word = y_stream.read();

    //     if (out_count < (unsigned)SEQ_LEN) {
    //         int t = out_count;  // one output word per time-step (assumption)
    //         for (int h = 0; h < HIDDEN_SIZE; ++h) {
    //             float y_hw  = (float)y_word[h];            // first 768 features
    //             float y_ref = (float)y_out_golden[t][h];   // golden
    
    //         // .. diff declaration

    //         }
    //     }
    //     out_count++;
    // }

}
