import numpy as np
import keras
import hls4ml
from hgq.layers import QMultiHeadAttention, QDense
from hgq.config import LayerConfigScope, QuantizerConfigScope

# ------------ Shape config ------------
BATCH    = 1
D        = 16 # model dimenstion (MHA embedding size), opt125m 768
head     = 2 #  head dimension, opt125m 12
D_head   = D // head # // make sure integer
T        = 1  # token (sequence length), this should be used for the regulation of HW
# -------------------------------------


# ----------------------------
# Build up the MHA, in future aligned with opt125m
# ----------------------------

# ignore training for current stage
input_form = keras.Input(shape=(T, D), name="mha_input") # name ="input" illegal

# Use default HGQ2 quantization scopes (same style as in HGQ2 README)
with (
    QuantizerConfigScope(place="all",     default_q_type="kbi", overflow_mode="SAT_SYM"),
    QuantizerConfigScope(place="datalane", default_q_type="kif", overflow_mode="WRAP"),
    LayerConfigScope(enable_ebops=True, beta0=1e-5),
):
    # Self-attention: query=value=input
    x = QMultiHeadAttention(
        num_heads=head,
        key_dim=D_head,
        name="mha",
    )(input_form, input_form) #Q=V (K defaults to V)

    # Optional: a small head after MHA (keeps everything HGQ2-style)
    x = QDense(D, name="dense_out")(x)

out = keras.layers.Flatten(name="flatten")(x)

k_model = keras.Model(inputs=input_form, outputs=out, name="hgq2_mha_only")

# Build the model by running one dummy forward
dummy = np.random.randn(BATCH, T, D).astype("float32")
_ = k_model(dummy)
k_model.summary()

# ------------------------------------------------------
# save model for neuron view and clean the output direcoty of hls4ml in advance
# -------------------------------------------------------
path_mha = "MHA.keras"
k_model.save(path_mha) # save in afile for extra neuron view

import os
import shutil
import stat

def make_writable_and_remove(path):
    # Make everything writable, then delete
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), stat.S_IWRITE)
        for f in files:
            os.chmod(os.path.join(root, f), stat.S_IWRITE)
    shutil.rmtree(path)

# ------------------------------------------------------
# hls4ml: Grab Configuration and convert through hls4ml 
# -------------------------------------------------------


# 1) Build an hls_config from the Keras model
hls_config = hls4ml.utils.config_from_keras_model(
    k_model,
    #input_shape=input_size, #
    granularity="name",   # per-layer config is possible
    default_precision= 'ap_fixed<16,6>', # presicion of data
    default_reuse_factor= 64 # repetition rounds of elements, should align with size configuration
)
# in MHA, Latency works for Vitis 
# Resource has too many restirction, check in future
hls_config['Model']['Strategy'] = 'Latency'

# # Optional: set global precision / reuse
# hls_config["Model"]["Precision"] = "ap_fixed<16,6>"
# hls_config["Model"]["ReuseFactor"] = 1

print("-----------------------------------")
print("Configuration of MHA module:")
print(hls_config)
print("-----------------------------------")

# 2) Convert to HLS

output_dir = "C:/Users/ziyuan.wang/hls4ml_MHA"
make_writable_and_remove(output_dir) # make reuse possible

hls_model = hls4ml.converters.convert_from_keras_model(
    k_model,
    hls_config=hls_config,
    output_dir=output_dir,
    project_name="MHA",
    backend="Vitis", # vivado is safer for MHA conversion
    io_type="io_parallel" # Heterogenous quantization for activations is only supported with IOType=io_parallel 
)
hls_model.write()
print("write finished")