"""Serve the final policy checkpoint with the chart tool interface."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from chartadept.io import ROOT, read_json


def command(args):
    protocol = read_json(ROOT / 'configs/inference.json')
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(args.model.resolve()), '--served-model-name', 'ChartAdept',
        '--host', args.host, '--port', str(args.port), '--dtype', 'bfloat16',
        '--max-model-len', str(protocol['max_model_len']),
        '--gpu-memory-utilization', str(args.memory_fraction),
        '--max-num-seqs', str(args.max_sequences), '--max-num-batched-tokens', '16384',
        '--enforce-eager', '--generation-config', 'vllm',
        '--limit-mm-per-prompt', json.dumps({'image': protocol['max_images_per_prompt']}),
        '--skip-mm-profiling', '--disable-uvicorn-access-log',
        '--enable-auto-tool-choice', '--tool-call-parser', 'hermes',
        '--mm-processor-kwargs', json.dumps(protocol['mm_processor_kwargs'])]


@contextmanager
def local_server(argv, output):
    """Start a local vLLM server for one invocation and stop it on exit."""
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    endpoint = f'http://127.0.0.1:{port}'
    output.mkdir(parents=True, exist_ok=True)
    log = output / 'serve.log'
    opener = build_opener(ProxyHandler({}))
    with log.open('w') as stream:
        server = subprocess.Popen([*argv, '--port', str(port)],
            stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'Inference server exited with status {server.returncode}. See {log}')
                try:
                    with opener.open(endpoint + '/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(f'Inference server did not become ready. See {log}')
            yield endpoint
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/gspo-step162')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8237)
    parser.add_argument('--memory-fraction', type=float, default=.80)
    parser.add_argument('--max-sequences', type=int, default=8)
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    argv = command(args)
    if args.print_command:
        print(json.dumps(argv, indent=2))
        return
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('VLLM_PORT', str(args.port + 100))
    os.execv(sys.executable, argv)


if __name__ == '__main__':
    main()
