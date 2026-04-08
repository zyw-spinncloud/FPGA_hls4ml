#include <iostream>
#include <cmath>
#include <direct.h>   // _chdir, _getcwd

#include "MLP_micro.h"                 // DUT
#include "defines.h"                   // input_t, result_t, fc*_weight_t, fc*_bias_t
#include "nnet_utils/nnet_helpers.h"   // nnet::load_weights_from_txt

int main() {
    // Change this to the folder that contains the "weights" folder
    const char* project_root =
        R"(C:\Users\Varsha.Ajith\Downloads\FPGA_hls4ml-main_MLP_micro_updated\FPGA_hls4ml-main\HLS4ML_full\MLP_micro_firmware)";

    if (_chdir(project_root) != 0) {
        std::perror("_chdir failed");
        return 1;
    
    }

    static const int W6_LEN    = 128;  // 1 * 2 * 64
    static const int B6_LEN    = 64;
    static const int W7_LEN    = 32;   // 1 * 2 * 16
    static const int B7_LEN    = 16;
    static const int PARAM_LEN = W6_LEN + B6_LEN + W7_LEN + B7_LEN; // 240

    // One inference input: 2 words of 16 values each
    static const int NUM_INPUT_WORDS = 2;

    // Streams
    hls::stream<input_t>  x_stream("x_stream");
    hls::stream<result_t> y_stream("y_stream");
    static model_default_t params[PARAM_LEN];

    // External weights 
    static fc1_weight_t w6[W6_LEN];
    static fc1_bias_t   b6[B6_LEN];
    static fc2_weight_t w7[W7_LEN];
    static fc2_bias_t   b7[B7_LEN];

    nnet::load_weights_from_txt<fc1_weight_t, W6_LEN>(w6,"w6.txt");
    nnet::load_weights_from_txt<fc1_bias_t, B6_LEN>(b6,"b6.txt");
    nnet::load_weights_from_txt<fc2_weight_t, W7_LEN>(w7,"w7.txt");
    nnet::load_weights_from_txt<fc2_bias_t, B7_LEN>(b7,"b7.txt");

    int idx = 0;

    for(int i=0;i<W6_LEN;i++)
        params[idx++] = (model_default_t)w6[i];

    for(int i=0;i<B6_LEN;i++)
        params[idx++] = (model_default_t)b6[i];

    for(int i=0;i<W7_LEN;i++)
        params[idx++] = (model_default_t)w7[i];

    for(int i=0;i<B7_LEN;i++)
        params[idx++] = (model_default_t)b7[i];
    
    if (idx != PARAM_LEN) {
        std::cerr << "Parameter packing error: idx = " << idx
                  << ", expected " << PARAM_LEN << std::endl;
        return 1;
    }

    // Create one sample input tensor of shape [2][16]
    // Since input_t has 16 elements, we write 2 input words.
    for (int t = 0; t < NUM_INPUT_WORDS; t++) {
        input_t in_word;
        for (int i = 0; i < 16; i++) {
            // Simple deterministic stimulus
            float val = 0.1f * (t * 16 + i + 1);
            in_word[i] = (ap_fixed<16,6>)val;
        }
        x_stream.write(in_word);
    }

    // Call DUT
    MLP_micro(x_stream, y_stream, params);

    // Read outputs
    int out_count = 0;
    while (!y_stream.empty()) {
        result_t out_word = y_stream.read();

        std::cout << "Output word " << out_count << ":\n";
        for (int i = 0; i < 64; i++) {
            std::cout << std::setw(3) << i << ": "
                      << std::setw(12) << (float)out_word[i] << "\n";
        }
        std::cout << "----------------------------------------\n";
        out_count++;
    }

    std::cout << "Done. Read " << out_count << " output word(s)." << std::endl;

    return 0;
}
