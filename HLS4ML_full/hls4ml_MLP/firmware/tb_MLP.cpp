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
            float hw = (float) out[h];                 // convert from ap_fixed
            float ref = y_out_golden[t][h];

            float diff = hw - ref;
            if (diff > 1e-2 || diff < -1e-2) {
                // print or assert; adjust tolerance depending on quantization
            }
        }
    }

    return 0;

    // // 4) Read out all results (io_stream) and compare to golden
    // unsigned out_count = 0;
    // float max_abs_err = 0.0f;
    // int   error_count = 0;

    // while (!y_stream.empty()) {
    //     result_t y_word = y_stream.read();

    //     if (out_count < (unsigned)SEQ_LEN) {
    //         int t = out_count;  // one output word per time-step (assumption)
    //         for (int h = 0; h < HIDDEN_SIZE; ++h) {
    //             float y_hw  = (float)y_word[h];            // first 768 features
    //             float y_ref = (float)y_out_golden[t][h];   // golden

    //             float err = std::fabs(y_hw - y_ref);
    //             if (err > max_abs_err) max_abs_err = err;
    //             if (err > 1e-2f) ++error_count;
    //         }
    //     }

    //     out_count++;
    // }

    // std::cout << "Observed " << out_count << " output words\n";
    // std::cout << "Max abs error = " << max_abs_err
    //           << ", error_count = " << error_count << std::endl;

    // if (error_count > 0) {
    //     std::cout << "TEST FAILED\n";
    //     return 1;
    // } else {
    //     std::cout << "TEST PASSED\n";
    //     return 0;
    // }
}
