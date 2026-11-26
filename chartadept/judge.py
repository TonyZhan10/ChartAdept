import asyncio,json,time
from pathlib import Path
from collections import Counter,deque
from datetime import datetime,timezone
import aiohttp
from chartadept.io import now,write_json,append_json
from chartadept.api import load_config,chat_payload,async_request_json
INTERNAL_RUBRIC = '''Grade the FINAL answer to a held-out synthetic scientific chart question.
The supplied ground-truth answer is for grading only. Do not invent an answer from the model's reasoning or from the ground truth when the final answer is absent or malformed.
If the model response is JSON with an answer field, extract that answer field; the reasoning field is not a substitute for the answer. An empty answer or a lone brace is incorrect even if the explanation mentions the right value.
For a categorical answer, require the requested exact label/legend/axis name or an unambiguous equivalent written form. Differences in whitespace, case, equivalent mathematical notation or punctuation are acceptable when they identify the SAME term; a distinct series or label is incorrect.
For a count, require the exact integer; an equivalent number written in words is acceptable.
For a numerical answer, require the same value at the precision requested by the QUESTION. Honor a stated rounding or decimal-place instruction; otherwise accept numerically equivalent written forms, not an arbitrary broad tolerance. Units may be reformatted only if the represented quantity and requested units are unchanged.
When multiple items are requested, all required items must be present and none may contradict the reference. Follow any requested ordering. A correct explanation with a wrong final answer receives zero.
Return a JSON object with score (integer 0 or 1), extracted_answer, and a brief explanation. Grade the final answer.
'''

class SharedJudge:
    """Rate-limit training reward requests and retries across runs."""

    def __init__(self, output, config, run_dirs, rpm=2000, concurrency=1024, retry_prompt_whitespace=False,
                 slow_request_hedge_seconds=0, stream_reviews=False):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = load_config(config)
        self.config_path=config
        self.run_dirs = {p.name: p for p in run_dirs}
        self.rpm = rpm
        self.concurrency = concurrency
        self.retry_prompt_whitespace = retry_prompt_whitespace
        self.slow_request_hedge_seconds = slow_request_hedge_seconds
        self.stream_reviews = stream_reviews
        self.failed_prompts = Counter()
        self.semaphore = asyncio.Semaphore(concurrency)
        self.start_lock = asyncio.Lock()
        self.request_times = deque()
        self.next_request = 0.0
        requests = self.output / 'judge-requests.jsonl'
        previous_requests = 0
        if requests.exists():
            utc, mono = datetime.now(timezone.utc), time.monotonic()
            for line in requests.open():
                previous_requests += 1
                age = (utc - datetime.fromisoformat(json.loads(line)['started_at'])).total_seconds()
                if 0 <= age < 60:
                    self.request_times.append(mono - age)
            if self.request_times:
                self.next_request = self.request_times[-1] + 60 / rpm
        self.grade_cache = {}
        self.grade_locks = {}
        cache = self.output / 'grade-cache.jsonl'
        if cache.exists():
            for line in cache.open():
                row = json.loads(line)
                self.grade_cache[row['prompt']] = row
        self.cached_grades = 0
        self.api_requests = previous_requests

    def recent_requests(self):
        cutoff = time.monotonic() - 60
        while self.request_times and self.request_times[0] <= cutoff:
            self.request_times.popleft()
        return len(self.request_times)

    async def request_start(self, context, attempt, max_tokens=None):
        # Space request starts under the lock, including after an event-loop delay.
        async with self.start_lock:
            self.recent_requests()
            delay = max(0, self.next_request - time.monotonic())
            if len(self.request_times) >= self.rpm:
                delay = max(delay, self.request_times[0] + 60 - time.monotonic())
            if delay:
                await asyncio.sleep(delay)
            self.recent_requests()
            started = time.monotonic()
            self.request_times.append(started)
            self.next_request = started + 60 / self.rpm
            self.api_requests += 1
            row = {'started_at': now(), 'attempt': attempt,
                   'reasoning_effort': self.config.get('reasoning_effort', 'low'),
                   'max_tokens': max_tokens,
                   'credential_revision': self.config.get('credential_revision'), **context}
            append_json(self.output / 'judge-requests.jsonl', row)
            append_json(self.run_dirs[context['run']] / 'judge-requests.jsonl', row)

    @staticmethod
    def validate_grade(parsed, score_key, answer_key):
        if not isinstance(parsed, dict):
            raise ValueError('Rubric response must be a JSON object')
        actual_answer_key = 'extract_answer' if answer_key == 'extracted_answer' and answer_key not in parsed else answer_key
        if parsed.get(score_key) not in (0, 1) or actual_answer_key not in parsed:
            raise ValueError('Rubric response missing binary score or extracted answer')

    async def request_copy(self, session, content, context, attempt, max_tokens, copy, suffix, grade_fields=None):
        actual_context = {**context, 'request_copy': copy}
        if self.retry_prompt_whitespace:
            actual_context['prompt_trailing_newlines'] = suffix
        await self.request_start(actual_context, attempt + 1, max_tokens)
        try:
            payload = chat_payload(content + '\n' * suffix if self.retry_prompt_whitespace else content,
                                   self.config, max_tokens)
            if self.stream_reviews:
                from chartadept.rl.sol_transport import streamed_review
                import uuid
                payload['stream'] = True
                raw = await streamed_review(session, payload, self.config,
                    self.output / 'stream-responses' / (uuid.uuid4().hex + '.json'),
                    timeout=self.config.get('timeout', 600), idle_timeout=180)
            else:
                raw = await async_request_json(session, 'POST', '/v1/chat/completions', payload,
                    timeout=self.config.get('timeout', 600), retries=0, config=self.config)
                import uuid
                write_json(self.output / 'nonstream-responses' / (uuid.uuid4().hex + '.json'), raw)
            body = raw['choices'][0]['message']['content'].strip()
            if body.startswith('```'):
                body = body.split('\n', 1)[1].rsplit('```', 1)[0].strip()
            parsed = json.loads(body)
            if grade_fields is not None:
                self.validate_grade(parsed, *grade_fields)
            return parsed, raw
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError, KeyError) as error:
            append_json(self.output / 'request-errors.jsonl', {**actual_context, 'attempt': attempt + 1,
                'error': f'{type(error).__name__}: {error}', 'updated_at': now()})
            raise

    async def judge(self, session, content, context, max_tokens=None, grade_fields=None):
        async with self.semaphore:
            if max_tokens is None:
                max_tokens = self.config.get('trajectory_audit_max_tokens', 4096)
            retries = self.config.get('retries', 0)
            suffix = self.failed_prompts[content] if self.retry_prompt_whitespace else 0
            for attempt in range(retries + 1):
                primary = asyncio.create_task(self.request_copy(session, content, context, attempt,
                    max_tokens, 0, suffix + attempt, grade_fields))
                tasks = {primary}
                try:
                    if not self.slow_request_hedge_seconds:
                        return await primary
                    done, _ = await asyncio.wait(tasks, timeout=self.slow_request_hedge_seconds)
                    if done:
                        return primary.result()
                    backup = asyncio.create_task(self.request_copy(session, content, context, attempt,
                        max_tokens, 1, suffix + attempt + 1, grade_fields))
                    tasks.add(backup)
                    append_json(self.output / 'hedge-events.jsonl', {**context, 'event': 'started',
                        'threshold_seconds': self.slow_request_hedge_seconds, 'updated_at': now()})
                    pending, errors = set(tasks), []
                    while pending:
                        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                        for task in sorted(done, key=lambda item: item is not primary):
                            try:
                                result = task.result()
                            except Exception as error:
                                errors.append(error)
                                continue
                            append_json(self.output / 'hedge-events.jsonl', {**context, 'event': 'winner',
                                'request_copy': int(task is backup), 'updated_at': now()})
                            return result
                    raise errors[-1]
                except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError, KeyError):
                    if attempt == retries:
                        if self.retry_prompt_whitespace:
                            self.failed_prompts[content] = suffix + attempt + len(tasks)
                        raise
                    await asyncio.sleep(2 + attempt)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    async def grade(self, session, prompt, context, score_key, answer_key, *, content=None):
        # Cache training rewards by their full prompt.
        lock = self.grade_locks.setdefault(prompt, asyncio.Lock())
        async with lock:
            cached = self.grade_cache.get(prompt)
            if cached:
                self.cached_grades += 1
                return cached['parsed'], cached['response'], cached['source_request']
            parsed, raw = await self.judge(session, prompt if content is None else content, context,
                max_tokens=self.config.get('answer_grading_max_tokens', 1024), grade_fields=(score_key, answer_key))
            row = {'prompt': prompt, 'parsed': parsed, 'response': raw,
                   'source_request': dict(context), 'updated_at': now()}
            append_json(self.output / 'grade-cache.jsonl', row)
            self.grade_cache[prompt] = row
            return parsed, raw, None
