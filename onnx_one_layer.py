#%%

from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers import BloomConfig
import torch
bloom_path = "bigscience/bloom"
model_dtype = torch.float16
config = BloomConfig.from_pretrained(bloom_path)
block = BloomBlock(config, layer_number=0)
block

#%%
from transformers.modeling_utils import _load_state_dict_into_model

# load a shard corresponding to layer model.transformer.h.0
path_to_shard = "/home/nicolas_huggingface_co/.cache/huggingface/transformers/5c67e76340d1ebec08c9d5fca744de06c91b81b31dcf8750c8c29ad8f025589e.578e7d60077fbfaeb530cbb1c04c6be3071183927f0706bdb92e1e8f07a32208"
state_dict = torch.load(path_to_shard)
# replace random weights with the ones from the shard
_load_state_dict_into_model(block, state_dict, "h.0.")
# cast block to device
block = block.eval().to(model_dtype).cuda(0)
#%%
# prepare dummy inputs
batch_size = 2
seq_len = 3
hidden_states = torch.randn(batch_size, seq_len, config.hidden_size).cuda(0).to(model_dtype) # [batch_size, seq_length, hidden_size]
layer_past = torch.empty(
                    2,
                    batch_size,
                    0,
                    config.n_head,
                    config.hidden_size // config.n_head,
                    device=hidden_states.device if type(hidden_states) is torch.Tensor else None
                ).to(model_dtype)

alibi = torch.randn(batch_size * config.n_head, 1, seq_len).cuda(0).to(model_dtype) # batch_size * n_head, 1, seq_len + past_seq_len
causal_mask = torch.ones(batch_size, 1, seq_len, seq_len).cuda(0).to(model_dtype) # batch_size, 1, seq_len + past_seq_len, seq_len

#%%
# test inference
with torch.no_grad():
    outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=None,
                    use_cache=False,
                    output_attentions=False,
                    alibi=alibi,
                )
outputs
#%%
# export to ONNX
from pathlib import Path
from torch.onnx import export as onnx_export
ONNX_SAVE_PATH = Path("tmp/test_onnx/h.0.onnx")
print("Generating:", ONNX_SAVE_PATH)
ONNX_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
device = next(block.named_parameters())[1].device
model_inputs = (hidden_states.to(device), layer_past.to(device), causal_mask.to(device), alibi.to(device), torch.tensor(False).to(device))
input_names = ["hidden_states", "layer_past", "attention_mask", "alibi", "use_cache"]
onnx_outputs = ["outputs"]  # ["hidden_states", "present", "attentions"]
dynamic_axes = {
    "hidden_states": {0: "batch_size", 1: "seq_len"},
    "layer_past": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 2: "seq_len", 3: "max_seq_len"},
    "alibi": {0: "batch_size * n_head", 2: "max_seq_len"},
}
onnx_export(
    block,
    model_inputs,
    f=ONNX_SAVE_PATH.as_posix(),
    input_names=input_names,
    output_names=onnx_outputs,
    dynamic_axes=dynamic_axes,
    do_constant_folding=True,  # removes None inputs from the graph
    opset_version=14,
    verbose=True,
)
# %%

# create ONNX session
import onnxruntime
import logging
logging.basicConfig(level=logging.DEBUG)
sess_options = onnxruntime.SessionOptions()
# Set graph optimization level
# sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_profiling = False
session = onnxruntime.InferenceSession(
                ONNX_SAVE_PATH.as_posix(),
                sess_options=sess_options,
                providers=[
                    ('TensorrtExecutionProvider', {
                        'device_id': 0,
                        # 'trt_max_workspace_size': 2147483648,
                        'trt_fp16_enable': True,
                    }),
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            # "enable_cuda_graph": '1'
                        },
                    ),
                    # "CPUExecutionProvider",
                ],
            )

# %%
# test parity
from io_binding_helper import inference_with_io_binding
onnx_outputs = inference_with_io_binding(session, hidden_states, layer_past, causal_mask, alibi, is_float16=True)
onnx_outputs

torch.testing.assert_close(outputs[0], onnx_outputs[0])
# %%
# test speed
import time
max_runs = 10
n_layers = config.n_layer

print("--- Pytorch ---")
timings = []
hs = hidden_states
with torch.no_grad():
    for i in range(max_runs):
        start = time.time()
        for _ in range(n_layers):
            outputs = block(
                        hs,
                        layer_past=layer_past,
                        attention_mask=causal_mask,
                        head_mask=None,
                        use_cache=False,
                        output_attentions=False,
                        alibi=alibi,
                    )
            hs = outputs[0]
        timings.append(time.time() - start)
print(f"Average time: {sum(timings) / max_runs:.2f} seconds")
print(f"Timings: {[f'{t:.2f}' for t in timings]}")


# %%
print("--- ONNX runtime ---")
timings = []
hs = hidden_states
for i in range(max_runs):
    start = time.time()
    for _ in range(n_layers):
        onnx_outputs = inference_with_io_binding(session, hs, layer_past, causal_mask, alibi, is_float16=True)
        hs = onnx_outputs[0]
    timings.append(time.time() - start)
print(f"Average time: {sum(timings) / max_runs:.2f} seconds")
print(f"Timings: {[f'{t:.2f}' for t in timings]}")
# %%
