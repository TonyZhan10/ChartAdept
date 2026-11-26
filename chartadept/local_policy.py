"""Local Qwen3-VL decoding with the chart tools and exact sampled action tokens."""
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

from chartadept.chat import terminal_answer
from chartadept.io import ROOT, read_json
from chartadept.tools import Environment, TOOLS


PROTOCOL = read_json(ROOT / 'configs/inference.json')
TOOL_NAMES = {tool['function']['name'] for tool in TOOLS}


def load_policy(checkpoint, device, *, training):
    # FP32 optimizer parameters retain the small GSPO updates between BF16 forwards.
    dtype = torch.float32 if training or device.type == 'cpu' else torch.bfloat16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, dtype=dtype, attn_implementation='sdpa', local_files_only=True).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    return model.eval(), processor


def encode_messages(processor, messages, tools=None, *, continuation=False):
    text = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    if continuation:
        # A sampled Qwen turn ends in im_end. The following newline belongs to the template.
        text = '\n' + text
    images = []
    for message in messages:
        if isinstance(message['content'], list):
            for part in message['content']:
                if part['type'] == 'image':
                    with Image.open(part['image']) as source:
                        images.append(source.convert('RGB'))
    return dict(processor(text=[text], images=images or None, return_tensors='pt',
                          images_kwargs=PROTOCOL['mm_processor_kwargs']))


def initial_context(processor, image, prompt, system=None):
    messages = [{'role': 'system', 'content': system or PROTOCOL['system_prompts']['zoom_python']},
                {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
    return encode_messages(processor, messages, TOOLS)


@dataclass
class Rollout:
    inputs: dict
    action_positions: list
    trace: dict


class PresencePenalty:
    def __init__(self, start, penalty):
        self.start = start
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        # Inference decodes one question at a time, with a fresh penalty for each assistant turn.
        scores[:, input_ids[0, self.start:].unique()] -= self.penalty
        return scores


def vision_input_token_ids(config):
    return [config.vision_start_token_id, config.vision_end_token_id, config.image_token_id, config.video_token_id]


def generate_turn(model, inputs, budget, *, sample):
    device = model.device
    model_inputs = {key: value.to(device) for key, value in inputs.items()}
    length = inputs['input_ids'].shape[1]
    config = GenerationConfig(do_sample=sample, max_new_tokens=budget, use_cache=True,
                              top_k=0 if sample else 50, top_p=1.0, temperature=1.0,
                              suppress_tokens=vision_input_token_ids(model.config),
                              eos_token_id=model.generation_config.eos_token_id,
                              pad_token_id=model.generation_config.pad_token_id)
    processors = [] if sample else [PresencePenalty(length, PROTOCOL['presence_penalty'])]
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        result = model.generate(**model_inputs, generation_config=config, use_model_defaults=False,
                                logits_processor=processors)
    return result[:, length:].cpu()


def parse_calls(text):
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
    if len(blocks) != text.count('<tool_call>') or len(blocks) != text.count('</tool_call>'):
        raise ValueError('Incomplete tool call')
    calls = []
    for block in blocks:
        call = json.loads(block)
        name, arguments = call['name'], call['arguments']
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError('Expected a tool name and an arguments object')
        calls.append((name, arguments))
    return calls


def execute_call(environment, name, arguments):
    try:
        if name not in TOOL_NAMES:
            raise ValueError('Unknown chart tool')
        return environment.execute(name, arguments)
    except (ValueError, TypeError, IndexError, KeyError, AttributeError) as error:
        return {'status': 'format_error', 'name': name, 'error': str(error), 'images': []}


def append_observations(inputs, observation):
    for key, values in observation.items():
        if key in inputs:
            dimension = 1 if key in ('input_ids', 'attention_mask') else 0
            inputs[key] = torch.cat((inputs[key], values), dim=dimension)
        else:
            inputs[key] = values


def rollout(model, processor, initial, image, directory, *, sample, max_tokens, context_limit):
    inputs = dict(initial)
    actions, turns, events = [], [], []
    answer = None
    tagged = False
    termination = 'assistant_turn_budget'
    environment = Environment([image], directory, sys.executable)
    try:
        for turn in range(1, PROTOCOL['agent_max_turns'] + 1):
            start = inputs['input_ids'].shape[1]
            budget = min(max_tokens - len(actions), context_limit - start)
            if budget <= 0:
                termination = 'context_budget' if start >= context_limit else 'completion_token_budget'
                break
            raw = generate_turn(model, inputs, budget, sample=sample)
            text = processor.tokenizer.decode(raw[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            count = raw.shape[1]
            actions.extend(range(start, start + count))
            inputs['input_ids'] = torch.cat((inputs['input_ids'], raw), dim=1)
            inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones_like(raw)), dim=1)
            turns.append({'turn': turn, 'text': text, 'input_tokens': start, 'input_images': len(inputs['image_grid_thw']),
                          'sampled_token_ids': raw[0].tolist()})
            for event in events:
                if 'observed_in_turn' not in event:
                    event['observed_in_turn'] = turn
            try:
                calls = parse_calls(text)
            except (ValueError, TypeError, KeyError) as error:
                turns[-1]['format_error'] = str(error)
                termination = 'malformed_tool_call'
                break
            if not calls:
                answer = terminal_answer({'content': text})
                tagged = bool(answer)
                if not sample and not tagged and count < budget:
                    answer = text.strip() or None
                termination = 'final_answer' if answer else ('completion_token_budget' if count == budget else 'missing_final_answer')
                break
            if turn == PROTOCOL['agent_max_turns']:
                break
            if len(events) + len(calls) > PROTOCOL['agent_max_tool_calls']:
                termination = 'tool_call_budget'
                break
            observations = []
            for name, arguments in calls:
                result = execute_call(environment, name, arguments)
                events.append({'after_turn': turn, 'name': name, 'arguments': arguments, 'result': result})
                observations.append({'role': 'tool', 'content': json.dumps(
                    {key: value for key, value in result.items() if key != 'images'}, ensure_ascii=False)})
                if result['images']:
                    observations.append({'role': 'user', 'content': [
                        {'type': 'text', 'text': 'Actual tool images returned by the environment. Inspect them and continue the original chart question.'},
                        *[{'type': 'image', 'image': path} for path in result['images']]]})
            if len(environment.images) > PROTOCOL['max_images_per_prompt']:
                termination = 'observed_image_budget'
                break
            observation = encode_messages(processor, observations, continuation=True)
            if inputs['input_ids'].shape[1] + observation['input_ids'].shape[1] >= context_limit:
                termination = 'context_budget'
                break
            append_observations(inputs, observation)
    finally:
        environment.close()
    trace = {'status': 'budget_exhausted' if termination.endswith('budget') else 'completed',
             'final_answer': answer, 'answer_tag_present': tagged, 'termination': termination, 'completion_tokens': len(actions),
             'tool_calls': len(events), 'successful_tools': sum(e['result']['status'] == 'success' for e in events),
             'turns': turns, 'tools': events}
    return Rollout(inputs, actions, trace)
