"""Grade terminal answers for GSPO using numerical rules and semantic judgments."""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import signal

from aiohttp import web, ClientSession, TCPConnector, ClientError
from chartadept.rl.answer_match import canonical, numeric_verdict
from chartadept.decimal_precision import POLICY, INSTRUCTION as DECIMAL_INSTRUCTION
from chartadept.reward.base import Broker
from chartadept.reward.reference_answers import reference_verdict
from chartadept.judge import SharedJudge, INTERNAL_RUBRIC
from chartadept import io as progress

PROTOCOL = 'chart-outcome-only-v4/' + POLICY
RUBRIC = INTERNAL_RUBRIC + '\nNumerical comparison policy: ' + DECIMAL_INSTRUCTION + '''\nPreserve signs: a positive magnitude is NOT equivalent to a negative change unless the question explicitly asks for magnitude. This request contains ONLY the final answer, not reasoning or tools; do not infer missing answers. Return score=1 for a correct complete answer, score=0 otherwise, extracted_answer and explanation.\n'''


def terminal_rule(answer, reference, question, answer_spec=None):
    answer = canonical(answer or '')
    if not answer or answer in ('{', '}', '{}', '[]'):
        return False, 'missing_or_malformed_terminal'
    if answer_spec:
        return reference_verdict(answer, answer_spec), 'verified_reference_answer_set'
    return numeric_verdict(answer, str(reference), question)


def outcome_reward(correct, metadata, source):
    """Use terminal correctness as the reward and record tool and token counts."""
    return {'score': float(correct), 'acc': float(correct), 'penalty': 0.0,
        'evidence_penalty': 0.0, 'tool_cost': 0.0, 'length_cost': 0.0,
        'zoom_calls': int(metadata['chart_zoom_calls']),
        'python_calls': int(metadata['chart_python_calls']),
        'generated_tokens': int(metadata['chart_generated_tokens']),
        'format_errors': int(metadata['chart_format_errors']),
        'execution_errors': int(metadata['chart_execution_errors']),
        'answer_verified': True, 'process_verified': False,
        'process_reward_enabled': False, 'grading_source': source}


class OutcomeBroker(Broker):
    reward_mode = 'outcome_only'
    judge_class = SharedJudge

    def __init__(self, output, config):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.options = json.loads(Path(config).read_text())
        assert self.options['reward_mode'] == self.reward_mode
        self.protocol = self.options.get('judge_protocol', PROTOCOL)
        if not self.protocol.endswith(POLICY):
            self.protocol += '/' + POLICY
        self.numeric_rewards = self.options.get('numeric_rewards', False)
        if self.numeric_rewards:
            self.protocol = 'numeric-terminal-answer/' + POLICY
        self.judge = None
        self.judge_model = 'numeric-reference' if self.numeric_rewards else self.options.get('model', 'gpt-6.1-sol')
        if not self.numeric_rewards:
            self.judge = self.judge_class(self.output / 'judge', Path(self.options['credential_config']), [self.output],
                rpm=self.options['rpm'], concurrency=self.options['concurrency'],
                retry_prompt_whitespace=True, slow_request_hedge_seconds=0, stream_reviews=True)
            self.judge.config.update(model=self.judge_model, reasoning_effort='low', temperature=0,
                timeout=600, retries=0, answer_grading_max_tokens=1024)
            if self.options.get('api_key_file'):
                self.judge.config.update(api_key=Path(self.options['api_key_file']).read_text().strip(),
                    base_url=self.options['base_url'], reasoning_effort=None,
                    chat_template_kwargs={'enable_thinking':False})
            elif self.judge_model == 'qwen3.8-27b':
                self.judge.config.update(reasoning_effort=None, chat_template_kwargs={'enable_thinking':False})
        self.rubric = ('Score the final numerical answer with the reference numeric rule. '
                       'A missing or unverified scalar scores zero.') if self.numeric_rewards else RUBRIC
        if self.judge_model == 'qwen3.8-27b':
            self.rubric += ('\nNumerical equivalence is not a printed-decimal-count check. Additional predicted decimal places are NOT a format error. '
                'For continuous values, use the bidirectional written-decimal rule above, rounding only one side at a time to the actual precision written by the other. '
                'Accept equality after that rounding; never reject solely because a prediction has extra digits. '
                'Counts, years, signs and distinct labels remain exact as described above.\n')
        import pyarrow.parquet as pq
        development = pq.read_table(self.options.get('development_file', 'data/rl/development.parquet'), columns=['extra_info']).to_pylist()
        self.development_ids = {row['extra_info']['question_id'] for row in development}
        assert len(self.development_ids) == self.options.get('development_qa', 256)
        self.counts, self.results, self.trace_locks, self.pending_retries = Counter(), {}, {}, {}
        self.session = None
        journal = self.output / 'rewards.jsonl'
        if journal.exists():
            for line in journal.open():
                row = json.loads(line)
                assert row['judge_protocol'] == self.protocol
                self.results[row['metadata']['chart_trace_file']] = row['reward']
                self.counts['completed'] += 1
        progress.write_json(self.output / 'judge-protocol.json', {'version': self.protocol, 'judge_model': self.judge_model,
            'reward': '1 if final answer is correct, else 0; no process/tool/length component',
            'rubric': self.rubric, 'api_input': ['question', 'reference_answer', 'model_final_answer'],
            'missing_api_score': 'pending and retry; never zero or partial K8',
            'process_reward_enabled': False, 'updated_at': progress.now()})

    def publish(self):
        progress.write_json(self.output / 'status.json', {'status': 'running', **dict(self.counts),
            'judge_model': self.judge_model, 'judge_role': 'final_answer_only', 'judge_protocol': self.protocol,
            'process_reward_enabled': False, 'waiting_retry': len(self.pending_retries),
            'rpm_limit': self.options['rpm'], 'concurrency': self.options['concurrency'],
            'actual_requests_last_minute': self.judge.recent_requests() if self.judge else 0,
            'actual_api_requests': self.judge.api_requests if self.judge else 0, 'updated_at': progress.now()})

    async def score_payload(self, payload):
        meta = payload['metadata']
        trace_file = meta['chart_trace_file']
        if trace_file in self.results:
            return web.json_response(self.results[trace_file])
        self.counts['active'] += 1
        self.publish()
        try:
            trace = await self.load_trace(trace_file)
            assert trace['id'] == payload['question_id']
            assert (trace['final_answer'] or '') == (meta['chart_final_answer'] or '')
            answer_spec = payload.get('answer_spec')
            correct, source = terminal_rule(meta['chart_final_answer'], payload['ground_truth'], payload['question'], answer_spec)
            if correct is None and self.numeric_rewards:
                correct, source = False, 'numeric_answer_required'
            semantic = correct is None
            if correct is None:
                grading_input = {
                    'question': payload['question'], 'reference_answer': payload['ground_truth'],
                    'model_final_answer': meta['chart_final_answer']}
                if answer_spec:
                    grading_input['verified_answer_specification'] = answer_spec
                prompt = self.rubric + '\nACTUAL GRADING INPUT:\n' + json.dumps(grading_input, ensure_ascii=False)
                if answer_spec:
                    prompt += ('\nThe verified specification resolves a known reference ambiguity. '
                        'For any_tied_maximum, any one or more valid_items is correct; do not require all. '
                        'For all_tied_maxima, require all and exclude extra items. '
                        'Use the verified specification when the reference lists a single answer.')
                context = {'run': self.output.name, 'id': payload['question_id'],
                    'method': meta['chart_route'], 'kind': 'rl_terminal_answer', 'trace_file': trace_file,
                    'judge_model': self.judge_model}
                parsed, _, cached = await self.judge.grade(self.session, prompt, context, 'score', 'extracted_answer')
                correct, source = parsed['score'] == 1, ('cached_' if cached else '') + self.judge_model
            reward = outcome_reward(correct, meta, source)
            reward = await self.complete_reward(payload, trace, reward)
            row = {'question_id': payload['question_id'], 'question': payload['question'],
                'ground_truth': payload['ground_truth'], 'metadata': meta, 'reward': reward,
                'judge_protocol': self.protocol, 'grading_source': source,
                'answer_spec': answer_spec,
                'data_split': 'development' if payload['question_id'] in self.development_ids else 'train',
                'updated_at': progress.now()}
            if reward.get('process_reward_enabled'):
                row['evidence_review'] = {'file': str(Path(trace_file).with_name('evidence-review.json'))}
            with (self.output / 'rewards.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            self.results[trace_file] = reward
            self.counts['completed'] += 1
            self.counts['semantic_completed' if semantic else 'rules_completed'] += 1
            if trace_file in self.pending_retries:
                self.pending_retries.pop(trace_file)
                self.counts['recovered_after_retry'] += 1
            return web.json_response(reward)
        except (ClientError, asyncio.TimeoutError, RuntimeError, ValueError, KeyError) as error:
            self.counts['infrastructure_errors'] += 1
            retryable = not str(error).startswith(tuple(f'Judge HTTP {code}:' for code in (400,401,403,404)))
            self.pending_retries[trace_file] = str(error)
            with (self.output / 'errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'trace_file': trace_file, 'error': str(error), 'updated_at': progress.now()}) + '\n')
            return web.json_response({'error': str(error), 'error_type': type(error).__name__,
                'retryable': retryable}, status=503 if retryable else 500)
        finally:
            self.counts['active'] -= 1
            self.publish()

    async def complete_reward(self, payload, trace, reward):
        return reward

    async def load_trace(self, path):
        return json.loads(Path(path).read_text())


async def run(args):
    broker = OutcomeBroker(args.output, args.config)
    broker.session = ClientSession(trust_env=broker.options.get('trust_env',True), connector=TCPConnector(limit=512))
    app = web.Application(client_max_size=8*1024*1024)
    app.router.add_post('/score', broker.score)
    app.router.add_get('/health', broker.health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', args.port)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    progress.write_json(args.output / 'broker-process.json', {'pid': os.getpid(),
        'url': f'http://{args.host}:{port}', 'judge_model': broker.judge_model,
        'judge_protocol': broker.protocol, 'started_at': progress.now()})
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        await broker.session.close()
        await runner.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
