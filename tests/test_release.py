import asyncio
import copy
from concurrent.futures import Executor, Future
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from chartadept.evaluate import extract_label, final_response, grade, prepare, summarize
from chartadept.infer import InferenceRunner
from chartadept.statistics import paired_bootstrap
from chartadept.tools import Environment


class CharacterTokenizer:
    """Deterministic tokenizer fixture for exercising exact window boundaries."""
    def encode(self, text, **kwargs):
        return list(text)

    def __call__(self, text, **kwargs):
        return {'input_ids': list(text), 'offset_mapping': [(i, i + 1) for i in range(len(text))]}

    def apply_chat_template(self, messages, **kwargs):
        return list(''.join(m['content'] for m in messages) + '<assistant>')


class ScriptedClient:
    """Protocol fixture; generated answers are not model evaluations."""
    def __init__(self, messages):
        self.messages = iter(messages)
        self.requests = []

    async def chat(self, messages, tools, choice, budget, seed, **kwargs):
        request = {'messages': copy.deepcopy(messages), 'tools': tools,
                   'tool_choice': choice, 'max_tokens': budget, **kwargs}
        self.requests.append(request)
        message = next(self.messages)
        finish = 'tool_calls' if message.get('tool_calls') else 'stop'
        raw = {'model': 'ChartAdept', 'choices': [{'message': message, 'finish_reason': finish}],
               'usage': {'completion_tokens': 3}}
        return {'message': message, 'request': request, 'response': raw,
                'finish_reason': finish, 'server': 'fixture://local'}


class InlineExecutor(Executor):
    """Run real tools synchronously in CPU protocol tests, without testing thread scheduling."""
    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


def tool_message(name, arguments, index):
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': f'call-{index}', 'type': 'function',
         'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


class EvaluationTests(unittest.TestCase):
    rubric = {'system_prompt': 'Grade the answer.',
              'user_template': 'Q={question}\nR={gold_answer}\nCandidate={llm_response}'}

    def test_long_response_is_not_cut_to_5000_characters(self):
        response = 'reasoning ' * 700 + '<answer>7</answer>'
        body, diagnostic = prepare(CharacterTokenizer(), self.rubric, 'Q', '7', response)
        self.assertIn(response, body['messages'][1]['content'])
        self.assertFalse(diagnostic['judge_input_truncated'])

    def test_native_limit_trims_prefix_without_breaking_unicode(self):
        response = '甲乙' * 300 + '<answer>7</answer>'
        body, diagnostic = prepare(CharacterTokenizer(), self.rubric, 'Q', '7', response,
                                   context=180, reserve=20)
        self.assertTrue(diagnostic['judge_input_truncated'])
        self.assertLessEqual(diagnostic['prompt_tokens'] + 20, 180)
        start, end = diagnostic['retained_character_range']
        self.assertTrue(body['messages'][1]['content'].endswith(response[start:end]))
        self.assertTrue(body['messages'][1]['content'].endswith('<answer>7</answer>'))

    def test_explicit_verdict_and_official_denominators(self):
        self.assertEqual(extract_label(r'Consider \boxed{A}. Final Judgment: \boxed{B} - INCORRECT'), 'B')
        self.assertIsNone(extract_label('A comparison is needed.'))
        questions = {'a': {'metric': 'one'}, 'b': {'metric': 'one'}, 'c': {'metric': 'two'}}
        result = summarize(questions, {'a': {'judge_label': 'A'}, 'c': {'judge_label': 'B'}})
        self.assertEqual(result['metrics']['one']['accuracy_percent'], 50)
        self.assertEqual(result['metrics']['one']['unresolved'], 1)
        self.assertEqual(result['macro_accuracy_percent'], 25)

    def test_full_last_message_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps({'turns': [
                {'response': {'content': 'Earlier turn'}},
                {'response': {'content': 'Evidence supports <answer>7</answer>', 'reasoning': 'Check the chart'}}]}))
            text = final_response({'trace_file': 'trace.json', 'final_answer': '7'}, directory)
            self.assertIn('Evidence supports', text)
            self.assertIn('Check the chart', text)
            self.assertNotIn('Earlier turn', text)

    def test_figure_draws_are_shared_across_metrics(self):
        questions, grades, baseline = {}, {}, {}
        for figure in (1, 2):
            for metric in ('a', 'b'):
                qid = f'{figure}-{metric}'
                questions[qid] = {'metric': metric, 'benchmark': 'toy', 'figure_id': figure}
                positive = (figure == 1) == (metric == 'a')
                grades[qid] = {'judge_label': 'A' if positive else 'B'}
                baseline[qid] = {'judge_label': 'B' if positive else 'A'}
        result = paired_bootstrap(questions, grades, baseline, samples=300, seed=17)
        self.assertEqual(result['macro_average']['percentile_95_ci_pp'], [0.0, 0.0])


class ToolAndLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / 'chart.png'
        chart = Image.new('RGB', (40, 20), 'red')
        chart.paste('blue', (20, 0, 40, 20))
        chart.save(self.image)
        self.question = {'id': 'fixture', 'task': 'custom', 'question': 'Read the chart.',
                         'prompt': 'Read the chart.', 'image': str(self.image),
                         'reference_answer': 'REFERENCE_MUST_NOT_REACH_THE_POLICY'}

    def tearDown(self):
        self.temp.cleanup()

    async def test_real_crop_and_python_in_the_interaction_loop(self):
        client = ScriptedClient([
            tool_message('image_zoom_in_tool', {'bbox_2d': [500, 0, 1000, 1000], 'img_idx': 0, 'label': 'right'}, 1),
            tool_message('image_code_tool', {'code': 'print(img.size[0] - 3)', 'img_idx': 1}, 2),
            {'role': 'assistant', 'content': '<answer>17</answer>'}])
        runner = InferenceRunner(self.root / 'run')
        runner.pool.shutdown(wait=True)
        runner.pool = InlineExecutor()
        try:
            trace = await runner.trajectory(client, self.question, 'zoom_python')
        finally:
            runner.pool.shutdown(wait=True)
        self.assertEqual(trace['status'], 'completed')
        self.assertEqual(trace['final_answer'], '17')
        self.assertEqual(trace['tool_calls'], 2)
        self.assertEqual(trace['tools'][1]['stdout'], '17\n')
        with Image.open(trace['tools'][0]['observation_image']) as crop:
            self.assertEqual(crop.size, (20, 20))
            self.assertEqual(crop.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual([x['observed_in_turn'] for x in trace['tools']], [2, 3])
        self.assertNotIn(self.question['reference_answer'], json.dumps(client.requests))

    async def test_autonomous_policy_can_answer_without_tools(self):
        client = ScriptedClient([{'role': 'assistant', 'content': '<answer>7</answer>'}])
        runner = InferenceRunner(self.root / 'direct-answer')
        runner.pool.shutdown(wait=True)
        runner.pool = InlineExecutor()
        try:
            trace = await runner.trajectory(client, self.question, 'zoom_python')
        finally:
            runner.pool.shutdown(wait=True)
        self.assertTrue(client.requests[0]['tools'])
        self.assertEqual(trace['tool_calls'], 0)
        self.assertEqual(trace['final_answer'], '7')

    async def test_answer_reserve_is_tool_free_and_within_total_budget(self):
        client = ScriptedClient([{'role': 'assistant', 'content': '<answer></answer>'},
                                 {'role': 'assistant', 'content': '{"answer":"7"}'}])
        runner = InferenceRunner(self.root / 'reserve', protocol_overrides={
            'total_completion_token_budget': 64, 'final_answer_reserve_tokens': 16,
            'structured_final_answer': True})
        runner.pool.shutdown(wait=True)
        runner.pool = InlineExecutor()
        try:
            trace = await runner.trajectory(client, self.question, 'zoom_python')
        finally:
            runner.pool.shutdown(wait=True)
        self.assertEqual(trace['final_answer'], '7')
        self.assertEqual(client.requests[0]['max_tokens'], 48)
        self.assertEqual(client.requests[1]['max_tokens'], 16)
        self.assertIsNone(client.requests[1]['tools'])
        self.assertEqual(client.requests[1]['tool_choice'], 'none')
        self.assertIn('response_format', client.requests[1])

    async def test_grading_transport_retains_full_response_and_missing_item(self):
        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            async def json(self):
                return {'model': 'fixture-judge', 'choices': [
                    {'message': {'content': r'Final Judgment: \boxed{A} - CORRECT'}}]}

        class Session(Response):
            def __init__(self, **kwargs):
                pass

            def post(self, *args, **kwargs):
                return Response()

        questions = {'one': {'question': 'Q', 'reference_answer': '7', 'metric': 'toy'},
                     'missing': {'question': 'Q2', 'reference_answer': '8', 'metric': 'toy'}}
        response = 'Evidence ' * 700 + '<answer>7</answer>'
        predictions = {'one': {'status': 'completed', 'full_response': response, 'final_answer': '7'}}
        args = SimpleNamespace(grades=self.root / 'grades.jsonl', predictions=self.root / 'index.jsonl',
            concurrency=1, attempts=1, judge_model='fixture-judge', judge_base_url='http://fixture.invalid/v1', timeout=1)
        with patch('aiohttp.ClientSession', Session):
            result = await grade(questions, predictions, args, CharacterTokenizer(), EvaluationTests.rubric)
        self.assertEqual(result['one']['score'], 1)
        self.assertEqual(result['missing']['judge_label'], 'UNRESOLVED')
        self.assertEqual(summarize(questions, result)['macro_accuracy_percent'], 50)
        request = json.loads((self.root / 'judge-requests.jsonl').read_text())
        self.assertIn(response, request['request']['messages'][1]['content'])


if __name__ == '__main__':
    unittest.main()
