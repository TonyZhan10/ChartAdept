"""Request a terminal-answer reward from the training reward broker."""
from decimal import Decimal, InvalidOperation
import asyncio
import json
import os
import re
import unicodedata

import aiohttp


def canonical(answer):
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False, sort_keys=True)
    return ' '.join(unicodedata.normalize('NFKC', str(answer)).split()).strip()


def exact_match(answer, reference):
    """Accept only exact text or exactly equivalent bare numbers; no tolerance."""
    left, right = canonical(answer), canonical(reference)
    if left == right:
        return True
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    if re.fullmatch(number, left) and re.fullmatch(number, right):
        try:
            return Decimal(left) == Decimal(right)
        except InvalidOperation:
            return False
    return False




async def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    # Grade the terminal answer in the metadata. The full rollout also contains tool observations.
    metadata = {key: extra_info[key] for key in ('chart_final_answer', 'chart_trace_file', 'chart_route',
        'chart_zoom_calls', 'chart_python_calls', 'chart_generated_tokens', 'chart_format_errors', 'chart_execution_errors',
        'chart_phase', 'chart_intervention')}
    payload = {'data_source': data_source, 'question_id': extra_info['question_id'],
               'question': extra_info['question'], 'ground_truth': ground_truth, 'metadata': metadata}
    if extra_info.get('answer_spec'):
        payload['answer_spec'] = extra_info['answer_spec']
    async with aiohttp.ClientSession(trust_env=False) as session:
        # Retry transient failures with the same trace so the broker can reuse a completed grade.
        connection_failures = 0
        sample_failures = 0
        while True:
            try:
                async with session.post(os.environ['CHART_RL_JUDGE_URL'].rstrip('/') + '/score', json=payload,
                                        timeout=aiohttp.ClientTimeout(total=900)) as response:
                    result = await response.json()
                    connection_failures = 0
                    if response.status == 200:
                        break
                    if result.get('sample_error'):
                        sample_failures += 1
                        if response.status == 422 or sample_failures >= 3:
                            from chartadept.reward.epoch_policy import SkipTrajectoryError
                            raise SkipTrajectoryError(result.get('error', 'Unscoreable sample'))
                    if response.status not in (429, 502, 503, 504) or result.get('retryable') is False:
                        raise RuntimeError(f'RL reward infrastructure failed: HTTP{response.status}, {result.get("error", "")[:300]}')
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                connection_failures += 1
                if connection_failures >= 8:
                    raise RuntimeError('RL reward broker is unreachable after eight attempts') from error
            await asyncio.sleep(15)
    values = {key: value for key, value in result.items() if isinstance(value, (int, float, bool))}
    if os.environ.get('CHART_RL_PAIRED_CONFIG'):
        present = bool(extra_info.get('chart_final_answer'))
        budget_ok = (extra_info['chart_generated_tokens'] <= 16384 and
            extra_info['chart_zoom_calls'] + extra_info['chart_python_calls'] <= 6)
        final_format = extra_info.get('chart_termination') == 'final_answer'
        values.update(raw_terminal_acc=values['acc'],
            effective_terminal_acc=values['acc'] * float(present and final_format and budget_ok),
            no_final_answer=float(not present), budget_invalid=float(not budget_ok),
            final_format_error=float(not final_format),
            total_tool_calls=extra_info['chart_zoom_calls']+extra_info['chart_python_calls'])
    return values
