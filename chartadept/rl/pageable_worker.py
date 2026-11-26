"""Use pageable host memory in vLLM when requested by the training configuration."""
import ray
from vllm.v1.worker.gpu_worker import Worker


class PageableWorker(Worker):
    def init_device(self):
        if __import__('os').environ.get('CHARTADEPT_PAGEABLE_BUFFERS') == '1':
            import vllm.v1.worker.gpu_model_runner as runner
            import vllm.device_allocator.cumem as allocator
            runner.is_pin_memory_available = lambda: False
            allocator.is_pin_memory_available = lambda: False
        return super().init_device()
