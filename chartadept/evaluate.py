"""Grade complete final responses with the 32K Compass protocol."""
import argparse
import asyncio
from collections import defaultdict
import json
import os
from pathlib import Path
import re

from chartadept.io import append_json, read_jsonl_by_id, write_json

POLICY = 'compass-cot-full-response-native32k-v1'


def final_response(prediction, folder):
    if 'full_response' in prediction:
        return prediction['full_response']
    path = Path(prediction['trace_file'])
    if not path.is_absolute():
        path = Path(folder) / path
    trace = json.loads(path.read_text())
    turn = trace['turns'][-1]
    message = turn.get('raw_api_response', {}).get('choices', [{}])[0].get('message')
    if message is None:
        message = turn.get('response')
    if message is None or 'content' not in message:
        raise ValueError(f'Missing complete final assistant response: {path}')
    text = message['content'] or ''
    if not isinstance(text, str):
        raise ValueError(f'Expected a text assistant response: {path}')
    reasoning = message.get('reasoning_content') or message.get('reasoning')
    if reasoning and reasoning not in text:
        text = '<think>\n' + reasoning + '\n</think>\n' + text
    return text


def prepare(tokenizer, rubric, question, reference, response,
            model='compass-verifier-7b', context=32768, reserve=2048):
    body = {'model': model, 'messages': [
        {'role': 'system', 'content': rubric['system_prompt']},
        {'role': 'user', 'content': ''}],
        'temperature': 1e-6, 'top_p': .9, 'top_k': 1,
        'max_tokens': reserve, 'stream': False}

    def render(candidate):
        body['messages'][1]['content'] = rubric['user_template'].format(
            question=question, gold_answer=reference, llm_response=candidate)
        return len(tokenizer.apply_chat_template(body['messages'], tokenize=True, add_generation_prompt=True))

    available = context - reserve - render('')
    if available < 1:
        raise ValueError('Question, reference, and rubric exceed the judge input budget.')
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    sent, start = response, 0
    input_tokens = render(sent)
    if input_tokens + reserve > context:
        cut = max(1, len(ids) - available)
        while True:
            start = offsets[cut][0] if cut < len(offsets) else len(response)
            sent = response[start:]
            input_tokens = render(sent)
            excess = input_tokens + reserve - context
            if excess <= 0:
                break
            cut = min(len(ids), cut + max(1, excess))
    return body, {'policy': POLICY, 'context_limit': context, 'output_reserve': reserve,
        'prompt_tokens': input_tokens, 'raw_response_tokens': len(ids),
        'sent_response_tokens': len(tokenizer.encode(sent, add_special_tokens=False)),
        'judge_input_truncated': start > 0, 'omitted_prefix_chars': start,
        'retained_character_range': [start, len(response)]}


def extract_label(text):
    text = text.strip()
    if text in ('A', 'B', 'C'):
        return text
    markers = list(re.finditer(r'Final\s+Judgment\s*:\s*', text, re.I))
    tail = text[markers[-1].end():] if markers else text
    boxed = re.findall(r'\\boxed\{\s*([ABC])\s*\}', tail)
    if boxed:
        return boxed[-1]
    if markers:
        match = re.match(r'([ABC])\b', tail)
        if match:
            return match[1]
    return None


def summarize(questions, grades):
    metrics = defaultdict(lambda: {'questions': 0, 'correct': 0, 'unresolved': 0})
    for qid, question in questions.items():
        entry = metrics[question['metric']]
        grade = grades.get(qid, {})
        entry['questions'] += 1
        entry['correct'] += grade.get('judge_label') == 'A'
        entry['unresolved'] += grade.get('judge_label') not in ('A', 'B', 'C')
    for entry in metrics.values():
        entry['accuracy_percent'] = 100 * entry['correct'] / entry['questions']
    return {'answer_policy': POLICY, 'metrics': dict(metrics),
        'questions': len(questions),
        'macro_accuracy_percent': sum(x['accuracy_percent'] for x in metrics.values()) / len(metrics)
        if metrics else None}


async def grade(questions, predictions, args, tokenizer, rubric):
    import aiohttp
    grades = read_jsonl_by_id(args.grades) if args.grades.exists() else {}
    slots = asyncio.Semaphore(args.concurrency)
    headers = {'Authorization': 'Bearer ' + os.environ['COMPASS_API_KEY']} if os.environ.get('COMPASS_API_KEY') else {}
    async with aiohttp.ClientSession(headers=headers, trust_env=False) as session:
        async def one(qid, question):
            if qid in grades:
                return
            row = {'id': qid, 'metric': question['metric'], 'answer_policy': POLICY}
            prediction = predictions.get(qid)
            if prediction is None or prediction.get('status') in ('infrastructure_error', 'retry_exhausted'):
                row.update(judge_label='UNRESOLVED', score=0, attempts=0, reason='No completed prediction')
            else:
                response = final_response(prediction, args.predictions.parent)
                body, diagnostics = prepare(tokenizer, rubric, question['question'],
                    question['reference_answer'], response, model=args.judge_model)
                row.update(judge_input=diagnostics, input_reference=question['reference_answer'])
                async with slots:
                    for attempt in range(1, args.attempts + 1):
                        try:
                            async with session.post(args.judge_base_url.rstrip('/') + '/chat/completions',
                                    json=body, timeout=aiohttp.ClientTimeout(total=args.timeout)) as result:
                                result.raise_for_status()
                                raw = await result.json()
                            append_json(args.grades.with_name('judge-requests.jsonl'),
                                        {'id': qid, 'attempt': attempt, 'request': body, 'response': raw})
                            if raw.get('model') != args.judge_model:
                                raise ValueError('Judge response uses a different model ID.')
                            content = raw['choices'][0]['message'].get('content') or ''
                            label = extract_label(content)
                            if label is None:
                                raise ValueError('Judge returned no explicit A/B/C verdict.')
                            row.update(judge_label=label, score=int(label == 'A'), attempts=attempt,
                                       explanation=content)
                            break
                        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, IndexError) as error:
                            if attempt == args.attempts:
                                row.update(judge_label='UNRESOLVED', score=0, attempts=attempt,
                                           reason=type(error).__name__)
                            else:
                                await asyncio.sleep(2)
            append_json(args.grades, row)
            grades[qid] = row
        await asyncio.gather(*(one(qid, q) for qid, q in questions.items()))
    return grades


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--grades', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--judge-base-url', default=os.environ.get('COMPASS_BASE_URL'))
    parser.add_argument('--judge-model', default='compass-verifier-7b')
    parser.add_argument('--judge-tokenizer', help='Compass tokenizer model ID or local directory.')
    parser.add_argument('--rubric', type=Path, help='JSON created by scripts/fetch_compass_rubric.py.')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--attempts', type=int, default=6)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    questions = read_jsonl_by_id(args.questions)
    if args.summarize_only:
        grades = read_jsonl_by_id(args.grades)
    else:
        if not all([args.predictions, args.judge_base_url, args.judge_tokenizer, args.rubric]):
            parser.error('Grading needs predictions, judge-base-url, judge-tokenizer, and rubric.')
        if min(args.concurrency, args.attempts, args.timeout) <= 0:
            parser.error('Concurrency, attempts, and timeout must be positive.')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.judge_tokenizer)
        rubric = json.loads(args.rubric.read_text())
        predictions = read_jsonl_by_id(args.predictions)
        args.grades.parent.mkdir(parents=True, exist_ok=True)
        grades = asyncio.run(grade(questions, predictions, args, tokenizer, rubric))
    result = summarize(questions, grades)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
