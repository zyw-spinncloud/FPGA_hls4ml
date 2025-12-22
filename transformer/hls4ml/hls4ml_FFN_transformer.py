import torch
import torch.nn as nn
import hls4ml
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import hls4ml_parameters as para

# ------------ Shape config ------------
BATCH = 1
H     = 16 # hidden size
T     = 1  # token (sequence length), this should be used for the regulation of HW
# -------------------------------------

# # input hf reference for weitghs loading
opt_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
opt_layer = opt_model.model.decoder.layers[0] # gran layer one as example

# # ----------------------------
# # Step: build up the FFN, transformer onw to test the embedding
# # ----------------------------

# since no training is needed here, 
# we ignore parameters like: dropout=0.1,activation_dropout=0.0

class FFN(nn.Module):
    def __init__(
        self,
        # vocab_size=32,
        hidden= H,         # hidden_size
        expand= 4,                 # ffn_dim = 4 * hidden_size = 3072
    ):
        super().__init__()

        d_MLP = expand * hidden  # 3072 FFN dimension for OPT-125M

        # No BatchNorm in OPT, just Linear layers.
        self.ln = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, d_MLP) # bias=True
        self.fc2 = nn.Linear(d_MLP, hidden) # bias=True
        self.act = F.relu # matching opt125m


    def forward(self, x):
        """
        x: (batch, seq_len, hidden) = (B, T, H)
        """
        # Pre-LayerNorm
        res = x
        # LayerNormalization
        y = self.ln(x)                           # (B, T, H)
        # First linear
        y = self.fc1(y)          # (B, T, 16 * 4)
        # # Nonlinearity
        y = self.act(y)          # ReLU in OPT-125M
        # non-linaer funtionaal + Second linear
        y = self.fc2(y)          # (B, T, 16)
        # Residual add
        x = res + y

        return x


# Create and initialize the model
model = FFN(hidden=H)
# input_size = (para.T, para.H) # this is used to definde input, output size
input_size = (T, H)

# ------------------------------------------------------
# hls4ml: Grab Configuration and convert through hls4ml 
# -------------------------------------------------------

model.eval()  # Set to evaluation mode
torch.save(model, "FFN_transformer.pt") # save in afile for extra neuron view

import os
print(os.path.abspath("FFN_transformer.pt"))

# Generate a configuration
config = hls4ml.utils.config_from_pytorch_model(
    model,
    input_shape=input_size, #
    default_precision='ap_fixed<16,6>', 
    granularity='name',
    default_reuse_factor= 4,
     channels_last_conversion='off' # important!!or else the array split error will occur
)

config['Model']['Strategy'] = 'Resource'

print("-----------------------------------")
print("Configuration of Py module:")
print(config)
print("-----------------------------------")

output_dir = "hls4ml_FFN_transformer"
# Convert the model to hls4ml
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model, 
    hls_config=config,
    output_dir=output_dir,
    project_name="FFN_transformer",
    backend='Vitis', #Catapult
    io_type='io_parallel'
)

hls_model.write()
print("write finished")


# # ---------------------------------------------------------------------
# # Test: Golden Output Generation:
# # ---------------------------------------------------------------------

# INPUT_SIZE = BATCH * para.T * para.H

# # -------------------------------------
# # 1. tokenizations and embedding (from outside)
# # sperate embedding layer is converted through keras-based model
# # Serve as the input data of model
# # -------------------------------------

# # 1) Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# # 2) Prepare some text
# text = "Hello, this is a test for the FFN block." # T=13
# tokens = tokenizer(text, return_tensors="pt")  
# print("Converted token looks like:")
# print(tokens)
# input_ids = tokens["input_ids"]  # shape: (batch=1, seq_len=T)

# # 3) Get embeddings from OPT's own embedding layer
# with torch.no_grad():
#     # This is the word embedding layer inside OPT
#     embed_tokens = opt_model.model.decoder.embed_tokens
#     x_embed = embed_tokens(input_ids)        # shape: (1, T, H)

# # ---------------------------------------------------------------------
# # 2. Run model to get outputs
# # ---------------------------------------------------------------------

# with torch.no_grad():
#     out = model(x_embed)                       # shape: (1, T, H)

# print("Input embedding shape:", x_embed.shape)
# print("Output shape:", out.shape)
# print("output result", out)

# # # -------------------------------------------
# # # detokenization: try to drive back to text
# # # -------------------------------------------

# # # convert to logits
# # lm_head = nn.Linear(para.H, tokenizer.vocab_size, bias=False)
# # lm_head.weight = opt_model.lm_head.weight
# # logits = lm_head(out)            # out is [1, 13, 16]

# # # convert to token ID
# # token_ids = torch.argmax(logits, dim=-1)     # [1, 13]

# # # detokenization
# # decoded_text = tokenizer.decode(token_ids[0])
# # print("Detokenized text is: ",decoded_text)



# # ---------------------------------------------------------------------
# # 3. Create the same deterministic pattern as the HLS testbench
# #    This matches what the HLS C testbench expects to read.
# # ---------------------------------------------------------------------

# def to_c_array(name, arr: np.ndarray) -> str:
#     """Convert a numpy array to a flattened C float array declaration."""
#     flat = arr.flatten()
#     s = f"const int {name}_SIZE = {flat.size};\n"
#     s += f"const float {name}[{flat.size}] = {{\n"
#     for i, v in enumerate(flat):
#         sep = "," if i < flat.size - 1 else ""
#         s += f"    {float(v):.8f}f{sep}\n"
#     s += "};\n\n"
#     return s

# def write_golden_header(x_embed, out, filename):
#     # Ensure on CPU and numpy
#     x_np = x_embed.detach().cpu().numpy()
#     y_np = out.detach().cpu().numpy()

#     assert x_np.shape == y_np.shape, "Input and output must have same shape"
#     B, para.T, para.H = x_np.shape  # e.g. 1, 13, 16

#     with open(filename, "w") as f:
#         f.write("#ifndef GOLDEN_DATA_H\n")
#         f.write("#define GOLDEN_DATA_H\n\n")

#         f.write("// Auto-generated golden output for FFN.\n\n")

#         # Shape macros
#         f.write(f"#define BATCH_SIZE {B}\n")
#         f.write(f"#define SEQ_LEN {para.T}\n")
#         f.write(f"#define HIDDEN_SIZE {para.H}\n\n")

#         # Write input tensor
#         f.write("static const float x_embed_golden[BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE] = {\n")
#         for b in range(B):
#             f.write("  {\n")
#             for t in range(para.T):
#                 f.write("    {")
#                 row = ", ".join(f"{x_np[b, t, h]:.8f}f" for h in range(para.H))
#                 f.write(row)
#                 f.write("}")
#                 if t != para.T - 1:
#                     f.write(",\n")
#                 else:
#                     f.write("\n")
#             f.write("  }")
#             if b != B - 1:
#                 f.write(",\n")
#             else:
#                 f.write("\n")
#         f.write("};\n\n")

#         # Write output tensor
#         f.write("static const float y_out_golden[BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE] = {\n")
#         for b in range(B):
#             f.write("  {\n")
#             for t in range(para.T):
#                 f.write("    {")
#                 row = ", ".join(f"{y_np[b, t, h]:.8f}f" for h in range(para.H))
#                 f.write(row)
#                 f.write("}")
#                 if t != para.T - 1:
#                     f.write(",\n")
#                 else:
#                     f.write("\n")
#             f.write("  }")
#             if b != B - 1:
#                 f.write(",\n")
#             else:
#                 f.write("\n")
#         f.write("};\n\n")

#         f.write("#endif // GOLDEN_DATA_H\n")


# GOLDEN_PATH = "hls4ml_FFN/firmware/FFN_golden_data.h"   # put it where HLS can see it

# write_golden_header(x_embed, out, GOLDEN_PATH)
# print(f"Golden data written to {GOLDEN_PATH}")