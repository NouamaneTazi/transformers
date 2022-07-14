# https://stackoverflow.com/questions/70740287/onnxruntime-inference-is-way-slower-than-pytorch-on-gpu
# http://www.xavierdupre.fr/app/onnxruntime/helpsphinx/api_summary.html#id4
import numpy
import torch
import logging
from typing import List, Dict, Union
from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


def get_output_buffers(output_shapes, device, is_float16=False):
    """Returns a dictionary of output name as key, and 1D tensor as value. The tensor has enough space for given shape."""
    data_type = torch.float16 if is_float16 else torch.float32

    output_buffers = {}
    for name, shape in output_shapes.items():
        output_buffers[name] = torch.empty(numpy.prod(shape), dtype=data_type, device=device)
    return output_buffers


def inference_with_io_binding(session, hidden_states, layer_number, layer_past, attention_mask, alibi):
    output_shapes = {"outputs": list(hidden_states.shape)}
    device_id = int(session.get_provider_options()["CUDAExecutionProvider"]["device_id"])
    device = torch.device(f"cuda:{device_id}")
    output_buffers = get_output_buffers(output_shapes, device)

    io_binding = IOBindingHelper.prepare_io_binding(
        session, hidden_states, layer_number, layer_past, attention_mask, alibi, output_buffers, output_shapes
    )

    session.run_with_iobinding(io_binding)

    outputs = IOBindingHelper.get_outputs_from_io_binding_buffer(
        session, output_buffers, output_shapes, return_numpy=False
    )
    return outputs


class TypeHelper:
    @staticmethod
    def get_input_type(ort_session: InferenceSession, name: str) -> str:
        for i, input in enumerate(ort_session.get_inputs()):
            if input.name == name:
                return input.type
        raise ValueError(f"input name {name} not found")

    @staticmethod
    def get_output_type(ort_session, name: str) -> str:
        for i, output in enumerate(ort_session.get_outputs()):
            if output.name == name:
                return output.type

        raise ValueError(f"output name {name} not found")

    @staticmethod
    def ort_type_to_numpy_type(ort_type: str):
        ort_type_to_numpy_type_map = {
            "tensor(int64)": numpy.longlong,
            "tensor(int32)": numpy.int32,  # numpy.intc?
            "tensor(float)": numpy.float32,
            "tensor(float16)": numpy.float16,
        }
        if ort_type not in ort_type_to_numpy_type_map:
            raise ValueError(f"{ort_type} not found in map")

        return ort_type_to_numpy_type_map[ort_type]

    @staticmethod
    def ort_type_to_torch_type(ort_type: str):
        ort_type_to_torch_type_map = {
            "tensor(int64)": torch.int64,
            "tensor(int32)": torch.int32,
            "tensor(float)": torch.float32,
            "tensor(float16)": torch.float16,
        }
        if ort_type not in ort_type_to_torch_type_map:
            raise ValueError(f"{ort_type} not found in map")

        return ort_type_to_torch_type_map[ort_type]

    @staticmethod
    def numpy_type_to_torch_type(numpy_type: numpy.dtype):
        numpy_type_to_torch_type_map = {
            numpy.longlong: torch.int64,
            numpy.int32: torch.int32,
            numpy.float32: torch.float32,
            numpy.float16: torch.float16,
        }
        if numpy_type not in numpy_type_to_torch_type_map:
            raise ValueError(f"{numpy_type} not found in map")

        return numpy_type_to_torch_type_map[numpy_type]

    @staticmethod
    def torch_type_to_numpy_type(torch_type: torch.dtype):
        torch_type_to_numpy_type_map = {
            torch.int64: numpy.longlong,
            torch.int32: numpy.int32,
            torch.float32: numpy.float32,
            torch.float16: numpy.float16,
        }
        if torch_type not in torch_type_to_numpy_type_map:
            raise ValueError(f"{torch_type} not found in map")

        return torch_type_to_numpy_type_map[torch_type]

    @staticmethod
    def get_io_numpy_type_map(ort_session: InferenceSession) -> Dict[str, numpy.dtype]:
        """Create a mapping from input/output name to numpy data type"""
        name_to_numpy_type = {}
        for input in ort_session.get_inputs():
            name_to_numpy_type[input.name] = TypeHelper.ort_type_to_numpy_type(input.type)

        for output in ort_session.get_outputs():
            name_to_numpy_type[output.name] = TypeHelper.ort_type_to_numpy_type(output.type)
        return name_to_numpy_type


class IOBindingHelper:
    @staticmethod
    def get_output_buffers(ort_session: InferenceSession, output_shapes, device):
        """Returns a dictionary of output name as key, and 1D tensor as value. The tensor has enough space for given shape."""
        output_buffers = {}
        for name, shape in output_shapes.items():
            ort_type = TypeHelper.get_output_type(ort_session, name)
            torch_type = TypeHelper.ort_type_to_torch_type(ort_type)
            output_buffers[name] = torch.empty(numpy.prod(shape), dtype=torch_type, device=device)
        return output_buffers

    @staticmethod
    def prepare_io_binding(
        ort_session,
        hidden_states,
        layer_number,
        layer_past,
        attention_mask,
        alibi,
        output_buffers,
        output_shapes,
        name_to_np_type=None,
    ):
        """Returnas IO binding object for a session."""
        if name_to_np_type is None:
            name_to_np_type = TypeHelper.get_io_numpy_type_map(ort_session)

        device_id = int(ort_session.get_provider_options()["CUDAExecutionProvider"]["device_id"])
        device = torch.device(f"cuda:{device_id}")

        # Bind inputs and outputs to onnxruntime session
        io_binding = ort_session.io_binding()

        # Bind inputs
        hidden_states = hidden_states.to(device)
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        io_binding.bind_input(
            "hidden_states",
            hidden_states.device.type,
            device.index,
            name_to_np_type["hidden_states"],
            list(hidden_states.size()),
            hidden_states.data_ptr(),
        )

        layer_number = layer_number.to(device)
        if not layer_number.is_contiguous():
            layer_number = layer_number.contiguous()
        io_binding.bind_input(
            "layer_number",
            layer_number.device.type,
            device.index,
            name_to_np_type["layer_number"],
            list(layer_number.size()),
            layer_number.data_ptr(),
        )

        layer_past = layer_past.to(device)
        if not layer_past.is_contiguous():
            layer_past = layer_past.contiguous()
        data_ptr = layer_past.data_ptr()
        if data_ptr == 0:
            # When past_sequence_length is 0, its data_ptr will be zero. IO Binding asserts that data_ptr shall not be zero.
            # Here we workaround and pass data pointer of input_ids. Actual data is not used for past so it does not matter.
            data_ptr = hidden_states.data_ptr()
        io_binding.bind_input(
            "layer_past",
            layer_past.device.type,
            device.index,
            name_to_np_type["layer_past"],
            list(layer_past.size()),
            data_ptr,
        )

        attention_mask = attention_mask.to(device)
        if not attention_mask.is_contiguous():
            attention_mask = attention_mask.contiguous()
        io_binding.bind_input(
            "attention_mask",
            attention_mask.device.type,
            device.index,
            name_to_np_type["attention_mask"],
            list(attention_mask.size()),
            attention_mask.data_ptr(),
        )

        alibi = alibi.to(device)
        if not alibi.is_contiguous():
            alibi = alibi.contiguous()
        io_binding.bind_input(
            "alibi", alibi.device.type, device.index, name_to_np_type["alibi"], list(alibi.size()), alibi.data_ptr()
        )

        # Bind outputs
        for output in ort_session.get_outputs():
            output_name = output.name
            output_buffer = output_buffers[output_name]
            logger.debug(f"{output_name} device type={output_buffer.device.type} shape={list(output_buffer.size())}")
            io_binding.bind_output(
                output_name,
                output_buffer.device.type,
                device.index,
                name_to_np_type[output_name],
                output_shapes[output_name],
                output_buffer.data_ptr(),
            )

        return io_binding

    @staticmethod
    def get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes, return_numpy=True):
        """Copy results to cpu. Returns a list of numpy array."""
        ort_outputs = []
        for output in ort_session.get_outputs():
            output_name = output.name
            buffer = output_buffers[output_name]
            shape = output_shapes[output_name]
            copy_tensor = buffer[0 : numpy.prod(shape)].reshape(shape).clone().detach()
            if return_numpy:
                ort_outputs.append(copy_tensor.cpu().numpy())
            else:
                ort_outputs.append(copy_tensor)
        return ort_outputs
