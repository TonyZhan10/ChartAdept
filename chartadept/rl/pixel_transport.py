"""Losslessly compress CPU float32 pixel buffers for transport and unpack each worker microbatch."""
from functools import wraps
import math
import os
import zlib

import torch


MODE = 'lossless-zlib-v1'
_counts = {'decoded_tensors': 0, 'decoded_raw_bytes': 0, 'decoded_wire_bytes': 0}


def pack_row(row):
    values = row.get('pixel_values')
    if values is None:
        return row
    assert values.device.type == 'cpu' and values.dtype == torch.float32
    packed = {'chart_pixel_transport': MODE, 'shape': tuple(values.shape),
              'buffer': zlib.compress(values.detach().contiguous().numpy().tobytes(), level=1)}
    return {**row, 'pixel_values': packed}


def unpack_row(row):
    if row is None:
        return row
    values = row.get('pixel_values')
    if not isinstance(values, dict) or values.get('chart_pixel_transport') != MODE:
        return row
    raw = zlib.decompress(values['buffer'])
    pixels = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(values['shape'])
    _counts['decoded_tensors'] += 1
    _counts['decoded_raw_bytes'] += len(raw)
    _counts['decoded_wire_bytes'] += len(values['buffer'])
    return {**row, 'pixel_values': pixels}


def status():
    return {'mode': MODE, 'pid': os.getpid(), **_counts}


def install():
    """Restore only the microbatch consumed by native actor/reference forward."""
    import verl.utils.model as model
    native = model.extract_multi_modal_inputs
    if getattr(native, '_chart_lossless_pixels', False):
        return

    @wraps(native)
    def extract(batch_data, indices=None):
        rows = batch_data if indices is None else [batch_data[i] for i in indices if i < len(batch_data)]
        return native([unpack_row(row) for row in rows])

    extract._chart_lossless_pixels = True
    model.extract_multi_modal_inputs = extract


def payload_statistics(batch):
    raw = wire = count = 0
    for row in batch.non_tensor_batch['multi_modal_inputs']:
        values = row['pixel_values']
        assert isinstance(values, dict) and values['chart_pixel_transport'] == MODE
        raw += math.prod(values['shape']) * 4
        wire += len(values['buffer'])
        count += 1
    return {'mode': MODE, 'rows': len(batch), 'packed_pixel_tensors': count,
            'raw_pixel_bytes': raw, 'wire_pixel_bytes': wire,
            'wire_fraction': wire / raw,
            'tensor_batch_bytes': sum(v.numel() * v.element_size() for v in batch.batch.values())}
