"""Native chart tools in verl, preserving sampled tokens and masking observations."""
import asyncio
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import time
from uuid import uuid4

from PIL import Image
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema

from chartadept.tools import Environment, TOOLS
from chartadept.chat import terminal_answer
from chartadept.io import write_json, now


def condition_messages(raw_prompt, intervention):
    """One public prompt builder for adaptive training and measured routes."""
    messages = deepcopy(list(raw_prompt))
    if isinstance(messages[0]['content'], list):
        assert all(part['type'] == 'text' for part in messages[0]['content'])
        messages[0]['content'] = ''.join(part['text'] for part in messages[0]['content'])
    for message in messages:
        if isinstance(message.get('content'), list):
            for part in message['content']:
                if part.get('type') == 'image':
                    part.update(min_pixels=802816, max_pixels=3211264)
    available = [] if intervention == 'direct' else TOOLS[:1] if intervention == 'crop' else TOOLS[1:] if intervention == 'python' else TOOLS
    if intervention in ('crop', 'python', 'crop_python'):
        instruction = {'crop': 'For this controlled route measurement, call image_zoom_in_tool before answering.',
                       'python': 'For this controlled route measurement, call image_code_tool before answering.',
                       'crop_python': 'For this controlled route measurement, first call image_zoom_in_tool, inspect its actual observation, then call image_code_tool, inspect its actual result, and answer.'}[intervention]
        messages[0]['content'] += ' ' + instruction
    elif intervention == 'direct':
        messages[0]['content'] += ' In this Direct condition no tools are available. Answer in this single turn from the original chart.'
    return messages, available


@register('chart_tools')
class ChartAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = ToolParser.get_tool_parser('hermes', self.tokenizer)
        self.max_turns = int(self.rollout_config.multi_turn.max_assistant_turns or 5)
        self.max_calls = 6
        spec = json.loads(Path(os.environ['CHART_RL_PAIRED_CONFIG']).read_text()) if os.environ.get('CHART_RL_PAIRED_CONFIG') else {}
        self.full_turn_budget = spec.get('per_turn_budget') == 'remaining_total'
        self.generated_budget = 16384
        self.response_length = int(self.rollout_config.response_length)
        self.skip_sample_failures = bool(spec.get('skip_sample_failures'))
        self.presence_penalty = spec.get('presence_penalty', 1.5)

    async def run(self, sampling_params, **kwargs):
        if not self.skip_sample_failures or not kwargs.get('chart_training_sample'):
            return await self._run(sampling_params, **kwargs)
        if kwargs.get('chart_epoch_padding'):
            return self._invalid_output(kwargs, 'epoch_tail_padding')
        from chartadept.reward.epoch_policy import sample_failure
        try:
            return await self._run(sampling_params, **kwargs)
        except Exception as error:
            if not sample_failure(error):
                raise
            return self._invalid_output(kwargs, type(error).__name__ + ': ' + str(error))

    def _invalid_output(self, kwargs, error):
        # Mark the failed or padded rollout for removal before reward attachment.
        request_id = uuid4().hex
        root = Path(os.environ['CHART_RL_ROLLOUT_ROOT']) / request_id
        root.mkdir(parents=True, exist_ok=True)
        record = {'id': kwargs['extra_info']['question_id'], 'request_id': request_id,
            'skip_trajectory': True, 'stage': 'generation', 'error': error,
            'reward_used': False, 'recorded_at': now()}
        write_json(root / 'runtime-error.json', record)
        if error != 'epoch_tail_padding' and not kwargs.get('chart_error_after_generation'):
            journal = root.parent.parent / 'trace-journals' / f'{socket.gethostname()}-{os.getpid()}.jsonl'
            journal.parent.mkdir(parents=True, exist_ok=True)
            with journal.open('a') as stream:
                stream.write(json.dumps({**record, 'phase': os.environ.get('CHART_RL_PHASE','cpu_replay'),
                    'finished_at': record['recorded_at']}) + '\n')
        eos = self.tokenizer.eos_token_id
        return AgentLoopOutput(prompt_ids=[eos], response_ids=[eos], response_mask=[0],
            response_logprobs=[0.], multi_modal_data={'images': None}, num_turns=0,
                metrics=AgentLoopMetrics(), extra_fields={'chart_runtime_invalid': record,
                    'chart_trace_file': str(root / 'runtime-error.json'), 'chart_final_answer': '',
                    'chart_phase': os.environ.get('CHART_RL_PHASE','cpu_replay'),
                    'chart_intervention': 'adaptive', 'chart_route': 'direct',
                    'chart_route_compliant': False, 'chart_tool_calls': 0, 'chart_zoom_calls': 0,
                    'chart_python_calls': 0, 'chart_generated_tokens': 0,
                    'chart_format_errors': 0, 'chart_execution_errors': 0,
                    'chart_termination': 'sample_infrastructure_error', 'chart_input_tokens': 1})

    async def _run(self, sampling_params, **kwargs):
        # Build the actor prompt from the chart, question, and tools for this condition.
        extra = kwargs.get('extra_info') or {}
        intervention = extra.get('route_intervention', 'adaptive')
        messages, available = condition_messages(kwargs['raw_prompt'], intervention)
        schemas = [OpenAIFunctionToolSchema(**tool) for tool in available]
        names = {tool['function']['name'] for tool in available}
        request_id = uuid4().hex
        root = Path(os.environ['CHART_RL_ROLLOUT_ROOT']) / request_id
        root.mkdir(parents=True, exist_ok=True)
        environment = Environment([extra['image_path']], root / 'tool-work', os.environ.get('CHART_RL_TOOL_PYTHON', os.sys.executable))
        vision = await self.process_vision_info(messages)
        images = list(vision.get('images') or [])
        prompt_ids = await self.apply_chat_template(messages, tools=available or None, images=images or None)
        initial_prompt = list(prompt_ids)
        response_mask, response_logprobs, events, turns = [], [], [], []
        metrics = {'generate_sequences': 0.0, 'tool_calls': 0.0, 'num_preempted': 0}
        generated = format_errors = execution_errors = 0
        have_logprobs = True
        final_answer, termination = None, 'assistant_turn_budget'
        began = time.monotonic()
        extra_fields = {'chart_runtime_invalid': None} if self.skip_sample_failures else {}
        try:
            for turn in range(1, 2 if intervention == 'direct' else self.max_turns + 1):
                remaining = min(self.generated_budget - generated, self.response_length - len(response_mask))
                if remaining <= 0:
                    termination = 'completion_token_budget'
                    break
                from chartadept.reward.protocol import generation_limit
                params = dict(sampling_params, max_tokens=generation_limit(remaining, intervention, self.full_turn_budget))
                params.setdefault('presence_penalty', self.presence_penalty)
                params.setdefault('repetition_penalty', 1.0)
                started = time.monotonic()
                input_tokens, input_images = len(prompt_ids), len(images)
                output = await self.server_manager.generate(request_id=request_id, prompt_ids=prompt_ids,
                    sampling_params=params, image_data=images or None, video_data=None)
                metrics['generate_sequences'] += time.monotonic() - started
                metrics['num_preempted'] += max(0, output.num_preempted or 0)
                extra_fields.update(output.extra_fields)
                raw_ids = list(output.token_ids)
                # Keep sampled token IDs for the policy loss and decode them for parsing and logging.
                prompt_ids.extend(raw_ids)
                response_mask.extend([1] * len(raw_ids))
                if output.log_probs is not None:
                    response_logprobs.extend(output.log_probs)
                else:
                    have_logprobs = False
                    response_logprobs.extend([0.0] * len(raw_ids))
                generated += len(raw_ids)
                for event in events:
                    if not event['sent_to_model']:
                        event.update(sent_to_model=True, observed_in_turn=turn)
                text = self.tokenizer.decode(raw_ids, skip_special_tokens=True)
                _, calls = await self.parser.extract_tool_calls(raw_ids, schemas)
                turns.append({'turn': turn, 'response': text, 'sampled_token_ids': raw_ids,
                              'input_tokens': input_tokens, 'input_images': input_images,
                              'presence_penalty': params['presence_penalty'],
                              'sampled_tokens': len(raw_ids), 'tool_calls': [call.model_dump() for call in calls]})
                if not calls:
                    if '<tool_call>' in text or '</tool_call>' in text:
                        format_errors += 1
                        termination = 'unparsed_tool_call'
                    else:
                        final_answer = terminal_answer({'content': text})
                        termination = 'final_answer' if final_answer else 'missing_final_answer'
                        format_errors += not bool(final_answer)
                    break
                if turn == self.max_turns:
                    termination = 'assistant_turn_budget'
                    break
                observation_messages, new_images = [], []
                for call in calls:
                    if len(events) >= self.max_calls:
                        termination = 'tool_call_budget'
                        break
                    started = time.monotonic()
                    event = {'index': len(events), 'after_turn': turn, 'name': call.name, 'raw_arguments': call.arguments,
                             'sent_to_model': False, 'status': 'Rejected', 'format_error': False}
                    try:
                        arguments = json.loads(call.arguments)
                        event['arguments'] = arguments
                        if call.name not in names:
                            raise ValueError('Unknown or unavailable tool: ' + call.name)
                        result = await asyncio.to_thread(environment.execute, call.name, arguments)
                        event.update(status='Success' if result['status'] == 'success' else 'ExecutionError',
                                     stdout=result.get('stdout', ''), stderr=result.get('stderr', ''),
                                     executed_code=arguments.get('code'), result=result)
                        execution_errors += result['status'] != 'success'
                    except (ValueError, KeyError, TypeError, IndexError) as error:
                        format_errors += 1
                        event.update(format_error=True, stderr=str(error))
                        result = {'status': 'format_error', 'name': call.name, 'error': str(error), 'images': []}
                    event['seconds'] = round(time.monotonic() - started, 6)
                    metrics['tool_calls'] += event['seconds']
                    events.append(event)
                    observation_messages.append({'role': 'tool', 'tool_call_id': f'{request_id}-{event["index"]}',
                        'content': json.dumps({key: value for key, value in result.items() if key != 'images'}, ensure_ascii=False)})
                    if result.get('images'):
                        event['observation_images'] = result['images']
                        observation_messages.append({'role': 'user', 'content': [
                            {'type': 'text', 'text': 'Actual tool images returned by the environment. Inspect them and continue the original chart question.'},
                            *[{'type': 'image', 'image': path, 'min_pixels': 802816, 'max_pixels': 3211264} for path in result['images']]]})
                if termination == 'tool_call_budget':
                    break
                vision = await self.process_vision_info(observation_messages)
                new_images = list(vision.get('images') or [])
                if len(images) + len(new_images) > 8:
                    termination = 'observed_image_budget'
                    break
                observation_ids = await self.apply_chat_template(observation_messages, images=new_images or None,
                                                                 remove_system_prompt=True)
                if len(response_mask) + len(observation_ids) >= self.response_length:
                    termination = 'observation_context_budget'
                    break
                prompt_ids.extend(observation_ids)
                response_mask.extend([0] * len(observation_ids))
                response_logprobs.extend([0.0] * len(observation_ids))
                images.extend(new_images)
        finally:
            await asyncio.to_thread(environment.close)
        counts = Counter(event['name'] for event in events)
        z, p = bool(counts['image_zoom_in_tool']), bool(counts['image_code_tool'])
        actual_route = 'crop_python' if z and p else 'crop' if z else 'python' if p else 'direct'
        compliant = intervention == 'adaptive' or actual_route == intervention
        if intervention == 'crop_python':
            compliant = compliant and any(z['name'] == 'image_zoom_in_tool' and p['name'] == 'image_code_tool' and
                                         z.get('observed_in_turn', 0) <= p['after_turn'] and z['after_turn'] < p['after_turn']
                                         for z in events for p in events)
        trace = {'id': extra['question_id'], 'request_id': request_id, 'route_intervention': intervention,
                 'phase': os.environ.get('CHART_RL_PHASE', 'cpu_replay'),
                 'actual_route': actual_route, 'route_compliant': compliant, 'input_image': extra['image_path'],
                 'prompt': {'system': messages[0]['content'], 'user': extra.get('question'), 'tools': available},
                 'policy_version': extra_fields,
                 'turns': turns, 'tools': events, 'final_answer': final_answer, 'termination': termination,
                 'completion_tokens': generated, 'tool_calls': len(events), 'format_errors': format_errors,
                 'execution_errors': execution_errors, 'seconds': time.monotonic() - began, 'finished_at': now()}
        write_json(root / 'trace.json', trace)
        journal = root.parent.parent / 'trace-journals' / f'{socket.gethostname()}-{os.getpid()}.jsonl'
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open('a') as stream:
            stream.write(json.dumps({key: trace[key] for key in ('id', 'request_id', 'phase', 'actual_route',
                'route_intervention', 'route_compliant', 'completion_tokens', 'tool_calls', 'format_errors',
                'execution_errors', 'seconds', 'finished_at')}) + '\n')
        response_ids = prompt_ids[len(initial_prompt):]
        assert len(response_ids) == len(response_mask) == len(response_logprobs)
        assert sum(response_mask) == generated
        if os.environ.get('CHART_RL_PHASE') == 'gpu_preflight':
            write_json(root / 'token-witness.json', {'prompt_ids': initial_prompt, 'response_ids': response_ids,
                        'response_mask': response_mask, 'response_logprobs': response_logprobs if have_logprobs else None,
                        'image_count': len(images)})
        extra_fields.update(chart_trace_file=str(root / 'trace.json'), chart_final_answer=final_answer or '',
                            chart_phase=os.environ.get('CHART_RL_PHASE', 'cpu_replay'), chart_intervention=intervention,
                            chart_route=actual_route, chart_route_compliant=compliant,
                            chart_tool_calls=len(events), chart_zoom_calls=counts['image_zoom_in_tool'],
                            chart_python_calls=counts['image_code_tool'], chart_generated_tokens=generated,
                            chart_format_errors=format_errors, chart_execution_errors=execution_errors,
                            chart_termination=termination, chart_input_tokens=len(initial_prompt))
        return AgentLoopOutput(prompt_ids=initial_prompt, response_ids=response_ids, response_mask=response_mask,
                               response_logprobs=response_logprobs if have_logprobs else None, multi_modal_data={'images': images},
                               num_turns=len(turns), metrics=AgentLoopMetrics(**metrics), extra_fields=extra_fields)
