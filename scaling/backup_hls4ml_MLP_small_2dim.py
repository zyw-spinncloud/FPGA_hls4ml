import torch
import torch.nn as nn
import hls4ml
import numpy as np

# ----------------------
# This is used to generate different configuration of hls4ml source file
# ------------ Shape config ------------
BATCH = 1
H     = 16 # hidden size
T     = 1  # token (sequence length), this should be used for the regulation of HW

# configuration to be discovered
REUSE = 16
PRECISION = 'ap_fixed<32,16>'
STRATEGY = 'Resource'
IO = 'io_stream'
# -------------------------------------


# ----------------------------
# Step1: MLP in small size 
# ----------------------------
import torch.nn.functional as F

class MLP_small_stream(nn.Module):
    def __init__(
        self,
        # vocab_size=32,
        hidden= H,         # hidden_size
        expand= 4,                 # ffn_dim = 4 * hidden_size = 3072
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
        x: (batch, seq_len, hidden) = (B, T, H)
        """
        y = self.fc1(x)          # (B, T, 16 * 4)
        y = self.act(y)          # ReLU in OPT-125M
        x = self.fc2(y)          # (B, T, 16)
        # x = res + y

        return x


# Create and initialize the model
model = MLP_small_stream(hidden=H)
input_size=(T, H) # this is used to definde input, output size


model.eval()  # Set to evaluation model
torch.save(model, "MLP_small_stream.pt") # save in afile for extra neuron view

import os
print(os.path.abspath("MLP_small_stream.pt"))

# ----------------------------
# Step2: Grab Configuration and convert through hls4ml 
# ----------------------------

# Generate a configuration
config = hls4ml.utils.config_from_pytorch_model(
    model,
    input_shape=input_size, # without the batch dim
    default_precision=PRECISION, # float-like precision 
    granularity='name',
    default_reuse_factor= REUSE
)
config['Model']['Strategy'] = STRATEGY

print("-----------------------------------")
print("Configuration of Py module:")
print(config)
print("-----------------------------------")

output_dir = f"hls4ml_MLP_small_stream"
# Convert the model to hls4ml
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model, 
    hls_config=config,
    output_dir=output_dir,
    project_name="MLP_small_stream",
    backend='Vitis', #Catapult
    io_type= IO
)


hls_model.write()
print("write finished")