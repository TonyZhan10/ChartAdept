"""Keep each TP1 vLLM executor's TCPStore outside client ephemeral ports."""
import os

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def port_base(cuda_visible_devices):
    # A node can host eight tensor-parallel-size-one replicas.
    return 26000 + 100 * int(cuda_visible_devices.split(',')[0])


class LowPortServer(vLLMHttpServer):
    def __init__(self, **kwargs):
        os.environ['VLLM_PORT'] = str(port_base(kwargs['cuda_visible_devices']))
        super().__init__(**kwargs)
