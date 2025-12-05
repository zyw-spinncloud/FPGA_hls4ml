import numpy as np
import keras
import hls4ml
from hgq.layers import QMultiHeadAttention, QDense
from hgq.config import LayerConfigScope, QuantizerConfigScope
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import OPTModel

# This is used to verify the functionality of MHA in the scale of opt125m
# Aiming to run for the C-sim

# ------------ Shape config ------------
BATCH    = 1
D        = 768 # model dimenstion (MHA embedding size)
head     = 12 #  head dimension
D_head   = D // head # // make sure integer
T        = 1  # token (sequence length), this should be used for the regulation of HW
# -------------------------------------

# -------------------------------------
# trained weights from hf-opt125m
# -------------------------------------

# input hf reference for weitghs loading
opt_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
opt_layer = opt_model.model.decoder.layers[0] # gran layer one as example
opt_attn = opt_layer.self_attn

# These are torch.nn.Linear modules, store them for later assignment
q_proj = opt_attn.q_proj
k_proj = opt_attn.k_proj
v_proj = opt_attn.v_proj
o_proj = opt_attn.out_proj

# ----------------------------
# Build up the MHA, in future aligned with opt125m
# ----------------------------


# import tensorflow as tf
# from tensorflow import keras

# class OptStyleQMHAAttention(keras.layers.Layer):
#     def __init__(self, num_heads, key_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.num_heads = num_heads
#         self.key_dim   = key_dim
#         self.proj_dim  = num_heads * key_dim

#         # these are QDense so they go through HGQ2 quantization
#         self.q_dense = QDense(self.proj_dim, use_bias=True, name="q_proj")
#         self.k_dense = QDense(self.proj_dim, use_bias=True, name="k_proj")
#         self.v_dense = QDense(self.proj_dim, use_bias=True, name="v_proj")
#         self.o_dense = QDense(self.proj_dim, use_bias=True, name="out_proj")

#     def _split_heads(self, x, batch_size):
#         # x: (B, T, D)
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))  # (B, T, H, d)
#         return tf.transpose(x, perm=[0, 2, 1, 3])  # (B, H, T, d)

#     def _combine_heads(self, x, batch_size):
#         # x: (B, H, T, d)
#         x = tf.transpose(x, perm=[0, 2, 1, 3])  # (B, T, H, d)
#         return tf.reshape(x, (batch_size, -1, self.proj_dim))  # (B, T, D)

#     def call(self, query, value, key=None, training=False):
#         # query, value, key: (B, T, D)
#         if key is None:
#             key = value

#         batch_size = tf.shape(query)[0]

#         # Linear projections with bias (OPT-style)
#         q = self.q_dense(query, training=training)  # (B, T, D)
#         k = self.k_dense(key,   training=training)  # (B, T, D)
#         v = self.v_dense(value, training=training)  # (B, T, D)

#         q = self._split_heads(q, batch_size)  # (B, H, T, d)
#         k = self._split_heads(k, batch_size)  # (B, H, T, d)
#         v = self._split_heads(v, batch_size)  # (B, H, T, d)

#         # scaled dot-product attention (OPT uses sqrt(d_k))
#         dk = tf.cast(self.key_dim, q.dtype)
#         scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(dk)  # (B, H, T, T)
#         attn   = tf.nn.softmax(scores, axis=-1)                   # (B, H, T, T)

#         context = tf.matmul(attn, v)           # (B, H, T, d)
#         context = self._combine_heads(context, batch_size)  # (B, T, D)

#         out = self.o_dense(context, training=training)  # OPT out_proj
#         return out


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
        use_bias=True, 
        name="mha",
    )(input_form, input_form) #Q=V (K defaults to V)

# model output is *just* the MHA result: (B, T, D)
k_model = keras.Model(inputs=input_form, outputs=x, name="hgq2_mha_only")

# Build the model by running one dummy forward
dummy = np.random.randn(BATCH, T, D).astype("float32")
_ = k_model(dummy)
k_model.summary()


# -------------------------
# read trained weights from hf to local model
# verify the correctness as well
# from torch to keras, need to build up some linaer 
# -------------------------

mha_layer = k_model.get_layer("mha") # traced by name
# print(mha_layer.__dict__.keys()) # help to check the real attribution name


import tensorflow as tf

print("query kernel:", mha_layer._query_dense.kernel.shape)
print("query bias:  ", mha_layer._query_dense.bias.shape)
print("q_proj weight:", q_proj.weight.shape)
print("q_proj bias:  ", q_proj.bias.shape)


def copy_linear_to_mha_kernel(pt_linear, keras_proj, name: str):
    """
    Copy a PyTorch nn.Linear into an HGQ2 QMultiHeadAttention projection.

    - pt_linear.weight: (out_features, in_features) = (768, 768)
    - pt_linear.bias:   (out_features,) = (768,) or may exist even if Keras has no bias

    - keras_proj.kernel: e.g. (768, 12, 64) 
    - keras_proj.bias:   may be None, or e.g. (12, 64)
    """
    with torch.no_grad():
        W_pt = pt_linear.weight.detach().cpu().numpy()  # (out, in)
        b_pt = pt_linear.bias.detach().cpu().numpy() if pt_linear.bias is not None else None

    # ----- kernel -----
    kernel_var   = keras_proj.kernel
    kernel_shape = kernel_var.shape  # e.g. (768, 12, 64)

    # Start from (in_features, out_features)
    W_in_out = W_pt.T  # (in, out) = (768, 768)

    if W_in_out.size != np.prod(kernel_shape):
        raise ValueError(
            f"[{name}] Incompatible kernel sizes: W_in_out.size={W_in_out.size}, "
            f"kernel_shape={kernel_shape}, product={np.prod(kernel_shape)}"
        )

    W_reshaped = W_in_out.reshape(kernel_shape)
    kernel_var.assign(W_reshaped.astype("float32"))

    # ----- bias (optional) -----
    bias_var = getattr(keras_proj, "bias", None)
    if bias_var is None:
        print(f"[{name}] keras_proj.bias is None, skipping bias copy")
    else:
        bias_shape = bias_var.shape  # e.g. (12, 64)
        if b_pt is None:
            print(f"[{name}] pt_linear has no bias, but keras expects one; setting zeros")
            b_reshaped = np.zeros(bias_shape, dtype="float32")
        else:
            if b_pt.size != np.prod(bias_shape):
                raise ValueError(
                    f"[{name}] Incompatible bias sizes: b_pt.size={b_pt.size}, "
                    f"bias_shape={bias_shape}, product={np.prod(bias_shape)}"
                )
            b_reshaped = b_pt.reshape(bias_shape).astype("float32")

        bias_var.assign(b_reshaped)

    print(f"[{name}] weights assigned; kernel_shape={kernel_shape}, "
          f"bias_shape={getattr(keras_proj, 'bias', None).shape if getattr(keras_proj, 'bias', None) is not None else None}")


def check_mha_kernel_matches_linear(keras_proj, pt_linear, name: str):
    """
    Verify that Keras kernel (+bias if present) match the PyTorch linear.
    """

    with torch.no_grad():
        W_pt = pt_linear.weight.detach()               # (out, in)
        b_pt = pt_linear.bias.detach() if pt_linear.bias is not None else None

    # ---- weights ----
    W_tf = keras_proj.kernel.numpy()                   # maybe 3D, e.g. (768, 12, 64)
    kernel_shape = W_tf.shape

    in_features  = W_pt.shape[1]                       # 768
    out_features = W_pt.shape[0]                       # 768

    # Flatten to (in, out)
    W_tf_in_out = W_tf.reshape(in_features, out_features)  # (768, 768)
    # Transpose to (out, in) to compare with PyTorch
    W_tf_out_in = torch.from_numpy(W_tf_in_out.T).to(W_pt.device)

    w_diff = (W_tf_out_in - W_pt).abs().max().item()
    print(f"{name}.weight max diff:", w_diff)

    # ---- bias ----
    bias_var = getattr(keras_proj, "bias", None)
    if bias_var is None or b_pt is None:
        print(f"{name}.bias   check skipped (keras or pt has no bias)")
    else:
        b_tf = bias_var.numpy()        # e.g. (12, 64)
        b_tf_flat = b_tf.reshape(-1)   # (768,)
        b_tf_pt   = torch.from_numpy(b_tf_flat).to(b_pt.device)
        b_diff = (b_tf_pt - b_pt).abs().max().item()
        print(f"{name}.bias   max diff:", b_diff)


# ---- 1) Copy weights into the 4 internal MHA projections ----
copy_linear_to_mha_kernel(q_proj, mha_layer._query_dense, name="q")
copy_linear_to_mha_kernel(k_proj, mha_layer._key_dense,   name="k")
copy_linear_to_mha_kernel(v_proj, mha_layer._value_dense, name="v")
copy_linear_to_mha_kernel(o_proj, mha_layer._output_dense,name="out")

q_dense = mha_layer._query_dense
k_dense = mha_layer._key_dense
v_dense = mha_layer._value_dense
o_dense = mha_layer._output_dense

# ---- 2) Check that the weights truly match ----
check_mha_kernel_matches_linear(q_dense, q_proj, "q")
check_mha_kernel_matches_linear(k_dense, k_proj, "k")
check_mha_kernel_matches_linear(v_dense, v_proj, "v")
check_mha_kernel_matches_linear(o_dense, o_proj, "out")


# -------------------------
# Functional alignment check
# -------------------------

# make sure functionality aligns
torch.manual_seed(0)
x_np = np.random.randn(BATCH, T, D).astype("float32")

x_tf = tf.convert_to_tensor(x_np)   # (B, T, D)
y_tf = mha_layer(x_tf, x_tf, training=False).numpy()  # (B, T, D)

device = next(opt_model.parameters()).device  # PyTorch device (cpu or cuda)
x_pt = torch.from_numpy(x_np).to(device)  # (B, T, D)

with torch.no_grad():
    # OPT self_attn API: hidden_states is (batch, seq_len, dim)
    # key_value_states=None => self-attention, no cross-attn
    y_pt, _ = opt_attn(
        hidden_states=x_pt,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        use_cache=False,
    )  # shape: (B, T, D)

y_pt_np = y_pt.cpu().numpy()

# ---- Compare ----
final_max_diff = np.max(np.abs(y_pt_np - y_tf))
print("MHA output max difference:", final_max_diff)
print("Keras MHA output shape:", y_tf.shape)
print("OPT MHA  output shape:", y_pt_np.shape)

# # ------------------------------------------------------
# # hls4ml: Grab Configuration and convert through hls4ml 
# # -------------------------------------------------------

# import os
# import shutil
# import stat

# def make_writable_and_remove(path):
#     # Make everything writable, then delete
#     if not os.path.exists(path):
#         return
#     for root, dirs, files in os.walk(path):
#         for d in dirs:
#             os.chmod(os.path.join(root, d), stat.S_IWRITE)
#         for f in files:
#             os.chmod(os.path.join(root, f), stat.S_IWRITE)
#     shutil.rmtree(path)


# # 1) Build an hls_config from the Keras model
# hls_config = hls4ml.utils.config_from_keras_model(
#     k_model,
#     #input_shape=input_size, #
#     granularity="name",   # per-layer config
#     default_precision= 'ap_fixed<16,6>', # presicion of data
#     default_reuse_factor= 64 # repetition rounds of elements, should align with size configuration
# )
# hls_config['Model']['Strategy'] = 'Latency'
# # Latency works for Vitis but Resource has too many restirction

# # # Optional: set global precision / reuse
# # hls_config["Model"]["Precision"] = "ap_fixed<16,6>"
# # hls_config["Model"]["ReuseFactor"] = 1

# print("-----------------------------------")
# print("Configuration of MHA_opt125m module:")
# print(hls_config)
# print("-----------------------------------")

# # 2) Convert to HLS

# output_dir = "C:/Users/ziyuan.wang/hls4ml_MHA_opt125m"
# make_writable_and_remove(output_dir) # make reuse possible

# hls_model = hls4ml.converters.convert_from_keras_model(
#     k_model,
#     hls_config=hls_config,
#     output_dir=output_dir,
#     project_name="MHA_opt125m",
#     backend="Vitis", # vivado is safer for MHA_opt125m conversion
#     io_type="io_parallel" # Heterogenous quantization for activations is only supported with IOType=io_parallel 
# )
# hls_model.write()
# print("write finished")


# # # -------------------------
# # weights check up, make sure after conversion still aligned
# # -------------------------

# print("fc1 weight is:",model.fc1.weight)
# print("fc1 weights' shape:",model.fc1.weight.shape)

# W = model.fc1.weight.detach().cpu().numpy().T # transpose needed at the end
# W_q = W.reshape(-1)
# # Print the first few
# print("flattened first 10:",W_q[:10])

# M = np.loadtxt("hls4ml_MLP/firmware/weights/w6.txt", delimiter=",") 
# print("Exported first 10 from w6:", M[:10])


# # ---------------------------------------------------------------------
# # Test: Golden Output Generation:
# # ---------------------------------------------------------------------

# INPUT_SIZE = BATCH * T * H

# # -------------------------------------
# # 1. tokenizations and embedding (from outside)
# # sperate embedding layer is converted through keras-based model
# # Serve as the input data of model,
# # independent emb layer is tested in another module
# # -------------------------------------

# # 1) Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# # 2) Prepare some text
# text = "0" # T=1, matching the MLP input size
# tokens = tokenizer(text, return_tensors="pt",add_special_tokens=False)
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
# # lm_head = nn.Linear(H, tokenizer.vocab_size, bias=False)
# # lm_head.weight = opt_model.lm_head.weight
# # logits = lm_head(out)            # out is [1, 13, 768]

# # # convert to token ID
# # token_ids = torch.argmax(logits, dim=-1)     # [1, 13]

# # # detokenization
# # decoded_text = tokenizer.decode(token_ids[0])
# # print("Detokenized text is: ",decoded_text)
