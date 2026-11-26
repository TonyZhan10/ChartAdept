"""Transport multimodal CPU tensors in VERL batches through Ray's NumPy buffers."""
from dataclasses import dataclass
import os
import socket

import numpy as np
import ray
import torch
from verl import DataProto


@dataclass
class _TensorFrame:
    values: np.ndarray
    dtype: torch.dtype
    shape: tuple


@dataclass
class _MultimodalFrames:
    rows: list
    shape: tuple


_installed = False
_serialized = 0
_deserialized = 0


def _pack(value):
    if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
        tensor = value.detach().contiguous()
        return _TensorFrame(tensor.reshape(-1).view(torch.uint8).numpy(), tensor.dtype, tuple(tensor.shape))
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pack(item) for item in value)
    return value


def _unpack(value):
    if isinstance(value, _TensorFrame):
        # Ray's NumPy views can be read-only. Own the receive buffer so verl
        # retains ordinary writable CPU tensors, including BF16 and integers.
        return torch.from_numpy(value.values.copy()).view(value.dtype).reshape(value.shape)
    if isinstance(value, dict):
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_unpack(item) for item in value)
    return value


def _serialize(data):
    global _serialized
    batch, non_tensor, meta = data.__getstate__()
    if 'multi_modal_inputs' in non_tensor:
        rows = non_tensor['multi_modal_inputs']
        non_tensor = dict(non_tensor)
        non_tensor['multi_modal_inputs'] = _MultimodalFrames([_pack(row) for row in rows.flat], rows.shape)
    _serialized += 1
    return batch, non_tensor, meta


def _deserialize(state):
    global _deserialized
    install()
    batch, non_tensor, meta = state
    if isinstance(non_tensor.get('multi_modal_inputs'), _MultimodalFrames):
        frames = non_tensor['multi_modal_inputs']
        rows = np.empty(len(frames.rows), dtype=object)
        for index, row in enumerate(frames.rows):
            rows[index] = _unpack(row)
        non_tensor = dict(non_tensor)
        non_tensor['multi_modal_inputs'] = rows.reshape(frames.shape)
    data = DataProto.__new__(DataProto)
    data.__setstate__((batch, non_tensor, meta))
    _deserialized += 1
    return data


def install():
    """Process-local, also installed when a peer decodes the new wire format."""
    global _installed
    if not _installed:
        ray.util.register_serializer(DataProto, serializer=_serialize, deserializer=_deserialize)
        _installed = True
    return status()


def status():
    return {'installed': _installed, 'serialized': _serialized, 'deserialized': _deserialized,
            'pid': os.getpid(), 'hostname': socket.gethostname(), 'protocol': 'chart-multimodal-numpy-v1'}


def install_in_actor(actor):
    """Use Ray's actor callback at a Python method boundary, without a restart."""
    return install()


def status_in_actor(actor):
    return status()
