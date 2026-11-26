import asyncio,json,time,sys
from pathlib import Path
from PIL import Image
from chartadept.chat import image,terminal_answer
from chartadept.tools import Environment,TOOLS
from chartadept import io as progress
ROOT=Path(__file__).resolve().parents[1]
TOOL_PYTHON=sys.executable
TOTAL_TOKENS=16384
METHODS={'zoom_python':'Zoom + Python'}
def relative(path):return str(Path(path).resolve())
def system_for(method):return json.loads((ROOT/'configs/inference.json').read_text())['system_prompt']
def tools_for(method):return [] if method=='tool_free' else TOOLS
def turn_completion_budget(method,remaining,experiment):return remaining
FINALIZE_MESSAGE = ('Complete the original chart question now using the original image and the actual observations already returned. '
                    'No more tools are available in this final response. Stop rechecking the same alternatives. '
                    'Give a brief, consistent conclusion covering every part of the question, with any required units, '
                    'and end with <answer>your complete answer</answer>. Do not repeat this instruction or invent tool observations.')
FINALIZE_JSON_MESSAGE = ('The available tool or reasoning budget is exhausted. Answer the original question using the original image '
                         'and actual observations already returned. No more tools are available. Return a JSON object with one '
                         'string field answer containing your complete answer, including every requested part and unit. '
                         'Give an answer, not a plan for further analysis. If the evidence is insufficient, answer Cannot determine. '
                         'Do not invent observations or emit tool calls.')
FINAL_ANSWER_FORMAT = {'type':'json_schema','json_schema':{'name':'chart_final_answer','strict':True,
    'schema':{'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}}}
async def trajectory(self,client,q,method):
        began=time.monotonic();folder=self.output/'traces'/q['id']/method;folder.mkdir(parents=True,exist_ok=True)
        previous=folder/'trace.json'
        if previous.exists() and json.loads(previous.read_text()).get('status')=='infrastructure_error':
            import shutil
            attempts=self.output/'infrastructure-attempts'/q['id']/method;attempts.mkdir(parents=True,exist_ok=True)
            archive=attempts/f'{len(list(attempts.iterdir()))+1:02d}'
            shutil.copytree(folder,archive)
            progress.write_json(archive/'artifact-location.json',{'original_folder':relative(folder),'archived_folder':relative(archive),'note':'Original request paths are unchanged in the saved trace; original artifacts are copied here before retry.'})
        actual_system=self.experiment_protocol.get('system_prompts',{}).get(method,system_for(method))
        messages=[{'role':'system','content':actual_system},
                  {'role':'user','content':[image(Path(q['image'])),{'type':'text','text':q['prompt']}]}]
        allowed=tools_for(method);names={t['function']['name'] for t in allowed}
        environment=Environment([Path(q['image'])],folder/'tool-work',TOOL_PYTHON)
        with Image.open(Path(q['image'])) as source:size=list(source.size)
        trace={'id':q['id'],'task':q['task'],'method':method,'model':self.experiment_protocol['model'],'status':'running','started_at':progress.now(),
            'tool_choice':self.experiment_protocol.get('tool_choice','auto'),
            'inference_mode':self.experiment_protocol.get('inference_mode','agent'),
            'question':q['question'],'input_image':q['image'],'input_model_image':q['image'],'input_size':size,
            'prompt':{'system':actual_system,'user':q['prompt'],'tools':allowed},'budget':self.experiment_protocol,
            'turns':[],'tools':[],'completion_tokens':0,'tool_calls':0,'successful_tools':0,'format_errors':0,
            'execution_errors':0,'final_answer':None,'trace_file':str((folder/'trace.json').relative_to(self.output))}
        self.save(trace,index=True)
        try:
            max_turns=self.experiment_protocol.get('direct_max_turns',12) if method=='tool_free' else self.experiment_protocol.get('agent_max_turns',12)
            reserve=self.experiment_protocol.get('final_answer_reserve_tokens',0)
            max_tools=self.experiment_protocol.get('agent_max_tool_calls',6)
            finalizing=False;recovery_reason=None
            for turn in range(1,max_turns+1):
                remaining=self.experiment_protocol.get('total_completion_token_budget',TOTAL_TOKENS)-trace['completion_tokens']
                if remaining<=0:trace['termination']='completion_token_budget';break
                if reserve and not finalizing and (recovery_reason or remaining<=reserve or trace['tool_calls']>=max_tools or turn==max_turns):
                    finalizing=True
                    reason=recovery_reason or ('tool_budget' if trace['tool_calls']>=max_tools else 'answer_reserve')
                    messages.append({'role':'user','content':FINALIZE_JSON_MESSAGE if self.experiment_protocol.get('structured_final_answer') else FINALIZE_MESSAGE})
                    trace.setdefault('recovery_events',[]).append({'turn':turn,'reason':reason,
                        'remaining_tokens':remaining,'tools_available':False,'reference_provided':False})
                budget=min(remaining,reserve) if finalizing else remaining-reserve if reserve else remaining
                structured=finalizing and self.experiment_protocol.get('structured_final_answer')
                final_kwargs={'response_format':FINAL_ANSWER_FORMAT,'sampling_overrides':{
                    'stop':[],'include_stop_str_in_output':False}} if structured else {}
                record=await client.chat(messages,tools=None if finalizing else allowed or None,
                    choice='none' if finalizing else trace['tool_choice'],
                    budget=turn_completion_budget(method,budget,self.experiment_protocol),seed=1729+turn-1,**final_kwargs)
                message=record['message'];calls=message.get('tool_calls') or [];usage=record['response'].get('usage',{})
                trace['completion_tokens']+=usage.get('completion_tokens',0)
                trace['turns'].append({'turn':turn,'request':record['request'],'response':message,'usage':usage,
                    'raw_api_response':record['response'],'finish_reason':record['finish_reason'],'server':record['server'],
                    'transport_errors':record.get('transport_errors',[])})
                if (trace['tool_choice']=='none' or finalizing) and calls:
                    raise RuntimeError('Backend returned a tool call with tool_choice=none')
                messages.append(message)
                # Mark the tool observations included in this completed request.
                for event in trace['tools']:
                    if not event.get('sent_to_model'):event.update(sent_to_model=True,observed_in_turn=turn)
                self.save(trace)
                if not calls:
                    content=message.get('content') or '';answer=terminal_answer(message)
                    if structured:
                        try:
                            answer=json.loads(content)['answer']
                        except json.JSONDecodeError:
                            if record['finish_reason']!='length':raise
                            trace['termination']='completion_token_budget'
                            trace['final_answer_format']='truncated_json_schema'
                            break
                        if not isinstance(answer,str) or not answer.strip():raise ValueError('Empty or non-text structured final answer')
                        trace['final_answer_format']='json_schema'
                    tool_text='<tool_call>' in content or '</tool_call>' in content
                    placeholder=(answer or '').strip().lower() in ['... your final answer (entity(s) or number) ...','your complete answer','your final answer']
                    if reserve and not finalizing and (placeholder or (answer is not None and not answer.strip())):
                        recovery_reason='placeholder_answer' if placeholder else 'empty_answer'
                        continue
                    if tool_text and answer is None:
                        trace['format_errors']+=1
                        if reserve and not finalizing:
                            recovery_reason='unparsed_tool_call'
                            continue
                        if method=='tool_free':
                            trace['unavailable_tool_requests']=trace.get('unavailable_tool_requests',0)+1
                            if trace['completion_tokens']>=self.experiment_protocol.get('total_completion_token_budget',TOTAL_TOKENS):
                                trace['termination']='completion_token_budget';break
                            messages.append({'role':'user','content':self.experiment_protocol['direct_unavailable_tool_response']})
                            self.save(trace)
                            continue
                        trace['termination']='unparsed_tool_call'
                    elif record['finish_reason']=='length' and answer is None:
                        if reserve and not finalizing:
                            recovery_reason='output_budget'
                            continue
                        trace['termination']='completion_token_budget'
                    elif not content.strip() and reserve and not finalizing:
                        recovery_reason='empty_response'
                        continue
                    else:
                        trace['final_answer']=answer if answer is not None else content.strip() or None
                        trace['answer_tag_present']=answer is not None;trace['termination']='final_answer'
                        if tool_text:
                            trace['format_errors']+=1
                            trace['unavailable_tool_text_with_explicit_terminal']=True
                    break
                for call in calls:
                    if trace['tool_calls']>=max_tools:
                        if reserve:
                            messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps({
                                'status':'not_executed','error':'The tool budget is exhausted; no additional tool was executed.'})})
                            trace.setdefault('unexecuted_tool_calls',[]).append(call)
                            recovery_reason='tool_budget'
                            continue
                        trace['termination']='tool_call_budget';break
                    index=trace['tool_calls'];name=call['function']['name'];arguments=call['function']['arguments'];start=time.monotonic()
                    event={'index':index,'after_turn':turn,'name':name,'raw_call':call,'raw_arguments':arguments,'status':'Rejected',
                        'stdout':'','stderr':'','format_error':False,'sent_to_model':False}
                    try:
                        arguments=json.loads(arguments) if isinstance(arguments,str) else arguments;event['arguments']=arguments
                        if name not in names:raise ValueError('Unknown or unavailable tool in this condition: '+name)
                        async with self.tool_slots:
                            result=await asyncio.get_running_loop().run_in_executor(self.pool,environment.execute,name,arguments)
                        event.update(result=result,status='Success' if result['status']=='success' else 'ExecutionError',
                                     stdout=result.get('stdout',''),stderr=result.get('stderr',''))
                        for key in ['source_image_index','source_size','source_bbox_pixels','output_image_index','output_image_indices','output_size','observation_transforms']:
                            if key in result:event[key]=result[key]
                        if 'source_bbox_pixels' in result:event['bbox_pixels']=result['source_bbox_pixels']
                        if 'output_size' in result:event['crop_size']=result['output_size']
                        images=result.get('images',[])
                        if images:
                            event.update(observation_image=relative(images[0]),observation_images=[relative(p) for p in images],native_crop=relative(images[0]))
                        event['executed_code']=arguments.get('code')
                    except (ValueError,KeyError,TypeError,IndexError) as error:
                        event.update(format_error=True,stderr=str(error));result={'status':'format_error','name':name,'error':str(error),'images':[]}
                    event['seconds']=round(time.monotonic()-start,3);trace['tools'].append(event);trace['tool_calls']+=1
                    trace['successful_tools']+=event['status']=='Success';trace['format_errors']+=event['format_error']
                    trace['execution_errors']+=event['status']!='Success' and not event['format_error']
                    messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps({k:v for k,v in result.items() if k!='images'},ensure_ascii=False)})
                    if result.get('images'):messages.append({'role':'user','content':[{'type':'text','text':'Actual tool images returned by the environment. Inspect them and continue the original chart question.'},*[image(p) for p in result['images']]]})
                    self.save(trace)
                if trace.get('termination')=='tool_call_budget':break
            trace.setdefault('termination','agent_turn_budget')
            trace['status']='completed' if trace['termination'] in ['final_answer','unparsed_tool_call'] else 'budget_exhausted'
        except Exception as error:
            trace.update(status='infrastructure_error',infrastructure_error=f'{type(error).__name__}: {error}')
            self.state['failed_requests']+=1
        finally:await asyncio.get_running_loop().run_in_executor(self.pool,environment.close)
        trace.update(seconds=round(time.monotonic()-began,3),messages=messages)
        self.save(trace,index=True)
        if trace['status'] in ['completed','budget_exhausted']:self.completed[(q['id'],method)]=self.summary(trace)
        self.state['completed_trajectories']=len(self.completed)
        return trace
