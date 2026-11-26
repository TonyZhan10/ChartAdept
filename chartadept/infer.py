"""Generate chart answers from a checkpoint or a vLLM endpoint."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import aiohttp

from chartadept._trajectory import trajectory
from chartadept.chat import NativeClient
from chartadept.checkpoints import require_checkpoint
from chartadept.io import ROOT, RUN_ROOT, append_json, now, output_directory, read_json, write_json
from chartadept.serve import local_server


class InferenceRunner:
    trajectory = trajectory

    def __init__(self, output, tool_choice='auto', mode='agent', protocol_overrides=None):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.experiment_protocol = read_json(ROOT / 'configs/inference.json')
        self.experiment_protocol['tool_choice'] = tool_choice
        self.experiment_protocol['inference_mode'] = mode
        if protocol_overrides:
            self.experiment_protocol.update(protocol_overrides)
        if mode == 'direct':
            direct = read_json(ROOT / 'configs/inference-tool-free.json')
            self.experiment_protocol.update(
                implementation_label=direct['implementation_label'], tool_choice='none',
                system_prompt=direct['system_prompt'], primary_method='tool_free',
                methods={'tool_free': 'Direct reasoning, no tools'},
                system_prompts={'tool_free': direct['system_prompt']}, tools={'tool_free': []},
                direct_unavailable_tool_response=direct['unavailable_tool_response'],
                direct_max_turns=direct['max_turns'])
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.tool_slots = asyncio.Semaphore(8)
        self.state = {'failed_requests': 0, 'completed_trajectories': 0}
        self.completed = {}

    def summary(self, trace):
        result = {key: trace.get(key) for key in (
            'id', 'method', 'task', 'status', 'updated_at', 'seconds',
            'completion_tokens', 'tool_calls', 'successful_tools', 'format_errors',
            'execution_errors', 'final_answer', 'termination', 'trace_file',
            'infrastructure_error', 'tool_choice', 'inference_mode', 'unavailable_tool_requests')}
        if 'recovery_events' in trace:
            result['recovery_events'] = trace['recovery_events']
        return result

    def save(self, trace, index=False):
        trace['updated_at'] = now()
        write_json(self.output / trace['trace_file'], trace)
        if index:
            append_json(self.output / 'index.jsonl', self.summary(trace))


async def generate_predictions(questions, output, server, concurrency=4, tool_choice='auto', mode='agent',
                               protocol_overrides=None):
    runner = InferenceRunner(output, tool_choice=tool_choice, mode=mode, protocol_overrides=protocol_overrides)
    protocol = runner.experiment_protocol
    sampling = {key: protocol[key] for key in (
        'temperature', 'top_p', 'top_k', 'presence_penalty', 'repetition_penalty',
        'stop', 'include_stop_str_in_output', 'stop_token_ids') if key in protocol}
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    results = []
    slots = asyncio.Semaphore(concurrency)
    try:
        async with aiohttp.ClientSession(connector=connector, trust_env=False) as session:
            servers = [server] if isinstance(server, str) else server
            client = NativeClient(session, [{'url': url.rstrip('/'), 'model': 'ChartAdept'} for url in servers],
                concurrency=concurrency, sampling=sampling,
                template_kwargs={'enable_thinking': False})

            async def one(question):
                question = dict(question)
                image = Path(question['image'])
                question['image'] = str(image if image.is_absolute() else ROOT / image)
                async with slots:
                    result = await runner.trajectory(client, question,
                        'tool_free' if mode == 'direct' else 'zoom_python')
                results.append(runner.summary(result))
                write_json(Path(output) / 'status.json', {
                    'status': 'running', 'requested': len(questions),
                    'tool_choice': protocol['tool_choice'], 'inference_mode': mode, **runner.state})

            await asyncio.gather(*(one(q) for q in questions))
    finally:
        runner.pool.shutdown(wait=True)
    write_json(Path(output) / 'status.json', {
        'status': 'completed' if not runner.state['failed_requests'] else 'needs_retry',
        'requested': len(questions), 'tool_choice': protocol['tool_choice'],
        'inference_mode': mode, **runner.state})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument('--server', action='append', help='vLLM endpoint; repeat for a server pool')
    backend.add_argument('--model', type=Path, help='Start and stop a local server for an exported model')
    parser.add_argument('--tool-choice', choices=('auto', 'none'), default='auto')
    parser.add_argument('--mode', choices=('agent', 'direct'), default='agent',
        help='direct: no tool schemas, explicit tool-free instructions, shared generation budget')
    parser.add_argument('--preset', choices=('main', 'museum', 'budget-control-32k'), default='main')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--question')
    parser.add_argument('--questions', type=Path, help='JSONL: id, image, question, prompt, task')
    parser.add_argument('--id', help='Select one ID from --questions')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--output', type=output_directory, default=str(RUN_ROOT / 'inference'))
    args = parser.parse_args()
    overrides = (None if args.preset == 'main' else
                 read_json(ROOT / f'configs/inference-{args.preset}.json'))
    checkpoint = None
    if args.model:
        try:
            checkpoint = require_checkpoint(args.model)
        except ValueError as error:
            parser.error(str(error))
        overrides = {**(overrides or {}), 'checkpoint': str(checkpoint)}
    if args.questions:
        questions = [json.loads(line) for line in args.questions.open()]
        if args.id:
            questions = [q for q in questions if q['id'] == args.id]
            if not questions:
                parser.error('Question ID not found')
    elif args.image and args.question:
        questions = [{'id': 'example', 'image': str(args.image.resolve()),
            'question': args.question, 'prompt': args.question, 'task': 'custom'}]
    else:
        parser.error('Supply --questions, or both --image and --question')
    if args.limit:
        questions = questions[:args.limit]
    # Resume only completed trajectories. A failed service request stays pending.
    pending = []
    for question in questions:
        method = 'tool_free' if args.mode == 'direct' else 'zoom_python'
        path = args.output / 'traces' / question['id'] / method / 'trace.json'
        if path.exists():
            trace = read_json(path)
            budget = trace.get('budget', {})
            if checkpoint and budget.get('checkpoint') != str(checkpoint):
                parser.error('Output contains a different model checkpoint; use a separate --output')
            choice = 'none' if args.mode == 'direct' else args.tool_choice
            if trace.get('tool_choice', 'auto') != choice or trace.get('inference_mode', 'agent') != args.mode:
                parser.error('Output contains a different tool-choice condition; use a separate --output')
            if budget.get('total_completion_token_budget') != (overrides or {}).get('total_completion_token_budget', 16384):
                parser.error('Output contains a different token budget; use a separate --output')
            if trace['status'] in ('completed', 'budget_exhausted'):
                continue
        pending.append(question)
    server = nullcontext(args.server or ['http://127.0.0.1:8237'])
    if checkpoint and pending:
        server = local_server([sys.executable, '-m', 'chartadept.serve',
            '--model', str(checkpoint), '--max-sequences', '1'], args.output)
    with server as endpoint:
        rows = asyncio.run(generate_predictions(pending, args.output, endpoint,
            args.concurrency, args.tool_choice, args.mode, overrides))
    for row in rows:
        print(json.dumps({'id': row['id'], 'answer': row['final_answer'],
            'tool_calls': row['tool_calls'], 'status': row['status']}, ensure_ascii=False))
    if any(row['status'] == 'infrastructure_error' for row in rows):
        sys.exit(1)


if __name__ == '__main__':
    main()
