import torch
import torch.nn as nn
import hls4ml
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# from transformers import OPTModel


# ------------ Shape config ------------
BATCH = 1
H     = 768 # hidden size
T     = 1  # token (sequence length), this should be used for the regulation of HW
# -------------------------------------

# -------------------------------------
# trained weights from hf-opt125m
# -------------------------------------

# # input hf reference for weitghs loading
opt_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
opt_layer = opt_model.model.decoder.layers[0] # gran layer one as example


# ----------------------------
# Build up the MLP, aligned with opt125m
# ----------------------------

# since no training is needed here, 
# we ignore parameters like: dropout=0.1,activation_dropout=0.0

class MLP(nn.Module):
    def __init__(
        self,
        # vocab_size=32,
        hidden=768,         # hidden_size
        expand=4,                 # ffn_dim = 4 * hidden_size = 3072
    ):
        super().__init__()

        d_MLP = expand * hidden  # 3072 FFN dimension for OPT-125M

        # No BatchNorm in OPT, just Linear layers.
        # self.emb = nn.Embedding(vocab_size, hidden) 
        self.fc1 = nn.Linear(hidden, d_MLP) # bias=True
        self.fc2 = nn.Linear(d_MLP, hidden) # bias=True
        self.act = F.relu # matching opt125m


    def forward(self, x):
        """
        x: (batch, seq_len, hidden) = (B, T, 768)
        """
        # x = self.emb(x) 
        # res = x

        # First linear
        y = self.fc1(x)          # (B, T, 3072)
        # # Nonlinearity
        y = self.act(y)          # ReLU in OPT-125M
        # non-linaer funtionaal + Second linear
        z = self.fc2(y)          # (B, T, 768)
        # # Residual add
        # x = res + y

        return z

# Create and initialize the model
model = MLP(hidden=H)
input_size=(None,H) # define clearly or else hls4ml conversion fails eg. (T,H) is impossible in this case

# for name, param in model.named_parameters():
#     print(name, param.device, param.dtype)

# -------------------------
# read trained weights from hf to local model
# verify the correctness as well
# -------------------------

# read the traiuned weights into the model
with torch.no_grad():
    model.fc1.weight.copy_(opt_layer.fc1.weight)
    model.fc1.bias.copy_(opt_layer.fc1.bias)
    model.fc2.weight.copy_(opt_layer.fc2.weight)
    model.fc2.bias.copy_(opt_layer.fc2.bias)

# verify correctness of weights and bias and results match
with torch.no_grad():
    print("fc1.weight max diff:", (model.fc1.weight - opt_layer.fc1.weight).abs().max().item())
    print("fc1.bias   max diff:", (model.fc1.bias   - opt_layer.fc1.bias).abs().max().item())
    print("fc2.weight max diff:", (model.fc2.weight - opt_layer.fc2.weight).abs().max().item())
    print("fc2.bias   max diff:", (model.fc2.bias   - opt_layer.fc2.bias).abs().max().item())

# make sure functionality aligns
torch.manual_seed(0)
x = torch.randn(1, T, H)

with torch.no_grad():
    opt_out = opt_layer.fc2(opt_layer.activation_fn(opt_layer.fc1(x)))
    my_out  = model(x)
print("Final max difference:", (opt_out - my_out).abs().max().item())


print("fc2's shape:",model.fc2.weight.shape)
# print("fc2 weight is:",model.fc2.weight)
# print("fc1 bias is:",model.fc1.bias)

# ------------------------------------------------------
# hls4ml: Grab Configuration and convert through hls4ml 
# -------------------------------------------------------

model.eval()  # Set to evaluation mode
torch.save(model, "MLP.pt") # save in afile for extra neuron view

import os
print(os.path.abspath("MLP.pt"))

# Generate a configuration
config = hls4ml.utils.config_from_pytorch_model(
    model,
    input_shape=input_size, #
    default_precision= 'ap_fixed<16,6>', # presicion of data
    granularity='name',
    default_reuse_factor= 64 # repetition rounds of elements, should align with size configuration
)
config['Model']['Strategy'] = 'Resource'
# layer configuration might be involved in the future

print("-----------------------------------")
print("Configuration of Py module:")
print(config)
print("-----------------------------------")

output_dir = "hls4ml_MLP"
# Convert the model to hls4ml
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model, 
    hls_config=config,
    output_dir=output_dir,
    project_name="MLP",
    backend='Vitis',
    io_type='io_parallel' #'io_stream'
)

hls_model.write()
print("write finished")


# -------------------------
# weights check up, make sure after conversion still aligned
# -------------------------

print("fc1 weight is:",model.fc1.weight)
print("fc1 weights' shape:",model.fc1.weight.shape)

W = model.fc1.weight.detach().cpu().numpy().T # transpose needed at the end
W_q = W.reshape(-1)
# Print the first few
print("flattened first 10:",W_q[:10])

M = np.loadtxt("hls4ml_MLP/firmware/weights/w6.txt", delimiter=",") 
print("Exported first 10 from w6:", M[:10])


# ---------------------------------------------------------------------
# Test: Golden Output Generation:
# ---------------------------------------------------------------------

INPUT_SIZE = BATCH * T * H

# -------------------------------------
# 1. tokenizations and embedding (from outside)
# sperate embedding layer is converted through keras-based model
# Serve as the input data of model,
# independent emb layer is tested in another module
# -------------------------------------

# 1) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# 2) Prepare some text
text = "0" # T=1, matching the MLP input size
tokens = tokenizer(text, return_tensors="pt",add_special_tokens=False)
print("Converted token looks like:")
print(tokens)
input_ids = tokens["input_ids"]  # shape: (batch=1, seq_len=T)

# 3) Get embeddings from OPT's own embedding layer
with torch.no_grad():
    # This is the word embedding layer inside OPT
    embed_tokens = opt_model.model.decoder.embed_tokens
    x_embed = embed_tokens(input_ids)        # shape: (1, T, H)

# ---------------------------------------------------------------------
# 2. Run model to get outputs
# ---------------------------------------------------------------------

with torch.no_grad():
    out = model(x_embed)                       # shape: (1, T, H)

print("Input embedding shape:", x_embed.shape)
print("Output shape:", out.shape)
print("output result", out)

# # -------------------------------------------
# # detokenization: try to drive back to text
# # -------------------------------------------

# # convert to logits
# lm_head = nn.Linear(H, tokenizer.vocab_size, bias=False)
# lm_head.weight = opt_model.lm_head.weight
# logits = lm_head(out)            # out is [1, 13, 768]

# # convert to token ID
# token_ids = torch.argmax(logits, dim=-1)     # [1, 13]

# # detokenization
# decoded_text = tokenizer.decode(token_ids[0])
# print("Detokenized text is: ",decoded_text)



# ---------------------------------------------------------------------
# 3. Create the same deterministic pattern as the HLS testbench
#    This matches what the HLS C testbench expects to read.
#    # io_parallel enabled, easier for c-sim
# ---------------------------------------------------------------------

def write_golden_header(x_embed, out, filename):
    """
    If in the future use io_stream:
      - BATCH_SIZE is assumed 1
      - stream dimension = SEQ_LEN
      - each stream word has HIDDEN_SIZE features
    """

    # move to CPU numpy
    x_np = x_embed.detach().cpu().numpy()
    y_np = out.detach().cpu().numpy()

    assert x_np.shape == y_np.shape, "Input and output must have same shape"
    B, T, H = x_np.shape  # e.g. 1, 1, 768
    assert B == 1, "This writer assumes BATCH_SIZE == 1"

    x_np = x_np[0]  # [T, H]
    y_np = y_np[0]  # [T, H]

    with open(filename, "w") as f:
        f.write("#ifndef GOLDEN_DATA_H\n")
        f.write("#define GOLDEN_DATA_H\n\n")

        f.write("// Auto-generated golden input/output for MLP (io_stream aligned).\n\n")

        # Shape macros
        f.write(f"#define BATCH_SIZE {B}\n")
        f.write(f"#define SEQ_LEN {T}\n")
        f.write(f"#define HIDDEN_SIZE {H}\n\n")

        # Input: [SEQ_LEN][HIDDEN_SIZE]
        f.write("static const float x_embed_golden[SEQ_LEN][HIDDEN_SIZE] = {\n")
        for t in range(T):
            f.write("  {")
            row = ", ".join(f"{x_np[t, h]:.8f}f" for h in range(H))
            f.write(row)
            f.write("}")
            if t != T - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # Output: [SEQ_LEN][HIDDEN_SIZE]
        f.write("static const float y_out_golden[SEQ_LEN][HIDDEN_SIZE] = {\n")
        for t in range(T):
            f.write("  {")
            row = ", ".join(f"{y_np[t, h]:.8f}f" for h in range(H))
            f.write(row)
            f.write("}")
            if t != T - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # Optional flattened versions
        x_flat = x_np.flatten()
        y_flat = y_np.flatten()

        f.write(f"static const int X_EMBED_GOLDEN_SIZE = {x_flat.size};\n")
        f.write(f"static const float x_embed_golden_flat[{x_flat.size}] = {{\n")
        for i, v in enumerate(x_flat):
            sep = "," if i < x_flat.size - 1 else ""
            f.write(f"    {float(v):.8f}f{sep}\n")
        f.write("};\n\n")

        f.write(f"static const int Y_OUT_GOLDEN_SIZE = {y_flat.size};\n")
        f.write(f"static const float y_out_golden_flat[{y_flat.size}] = {{\n")
        for i, v in enumerate(y_flat):
            sep = "," if i < y_flat.size - 1 else ""
            f.write(f"    {float(v):.8f}f{sep}\n")
        f.write("};\n\n")

        f.write("#endif // GOLDEN_DATA_H\n")


GOLDEN_PATH = "hls4ml_MLP/firmware/mlp_golden_data.h"   # put it where HLS can see it

write_golden_header(x_embed, out, GOLDEN_PATH)
print(f"Golden data written to {GOLDEN_PATH}")