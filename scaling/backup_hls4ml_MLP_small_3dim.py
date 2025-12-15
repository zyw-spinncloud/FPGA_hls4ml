import torch
import torch.nn as nn
import hls4ml
import numpy as np

# ----------------------
# This is used to generate different configuration of hls4ml source file
# ------------ Shape config ------------
BATCH = 1
C     = 4 # channel
H     = 4  # height
W     = 4  # width
E     = 4 # expand

# configuration to be discovered
REUSE = 16
PRECISION = 'ap_fixed<32,16>'  # 'ap_fixed<16,6>'
STRATEGY = 'Latency'    # 'Resource'
IO = 'io_parallel'  #'io_stream'
# -------------------------------------


# ----------------------------
# Step1: MLP in small size 
# ----------------------------
import torch.nn.functional as F

class MLP_small(nn.Module):
    def __init__(self, num_channels, expand=4):
        super().__init__()

        # Channel MLP_small (1x1 convs)
        hidden = expand * num_channels
        self.bn2   = nn.BatchNorm2d(num_channels)
        self.mlp1  = nn.Conv2d(num_channels, hidden, kernel_size=1, bias=True)
        self.act2  = nn.ReLU(inplace=True)
        self.mlp2  = nn.Conv2d(hidden, num_channels, kernel_size=1, bias=True)

    def forward(self, x):
        z = x
        # channel MLP_small 
        y = self.bn2(x)
        y = self.mlp1(y)
        y = self.act2(y)
        y = self.mlp2(y)
        x = z + y                                   # residual add
        return x
    

# Create and initialize the model
model = MLP_small(num_channels=C)
input_size=(C, H, W)


model.eval()  # Set to evaluation model
torch.save(model, "MLP_small.pt") # save in afile for extra neuron view

import os
print(os.path.abspath("MLP_small.pt"))

# ----------------------------
# Step2: Grab Configuration and convert through hls4ml 
# ----------------------------

# Generate a configuration
config = hls4ml.utils.config_from_pytorch_model(
    model,
    input_shape=input_size, # (C, H, W) without the batch dim
    default_precision=PRECISION, # float-like precision 
    granularity='name',
    default_reuse_factor= REUSE
)
config['Model']['Strategy'] = STRATEGY

print("-----------------------------------")
print("Configuration of Py module:")
print(config)
print("-----------------------------------")

output_dir = f"hls4ml_MLP_small_{IO}_{STRATEGY}_{REUSE}_{PRECISION}"
# Convert the model to hls4ml
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model, 
    hls_config=config,
    output_dir=output_dir,
    project_name="MLP_small",
    backend='Vitis', #Catapult
    io_type= IO
)


hls_model.write()
print("write finished")