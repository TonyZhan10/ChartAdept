"""Send multimodal chat requests to vLLM."""
import asyncio
from collections import deque
from decimal import Decimal
from datetime import datetime,timezone
from functools import lru_cache
import json
from pathlib import Path
import re
import time

import aiohttp
import msgspec

from chartadept.tools import image_content
from chartadept import io as progress


def image(path):
    return {'type':'image','image':str(Path(path).resolve())}


@lru_cache(maxsize=1024)
def encoded(path):
    return image_content(path)


def wire(messages):
    return [{**m,'content':[encoded(c['image']) if c.get('type')=='image' else c for c in m['content']]}
            if isinstance(m.get('content'),list) else m for m in messages]


def native_body(payload):
    return msgspec.json.encode({**payload,'messages':wire(payload['messages'])})


def parsed_json(content):
    content=(content or '').strip()
    if content.startswith('```'):content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
    return json.loads(content)


def terminal_answer(message):
    text=message.get('content') or ''
    match=re.findall(r'<answer>(.*?)</answer>',text,re.S)
    return match[-1].strip() if match else None


def append(path,row):
    with Path(path).open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')


class RequestGate:
    """Observable request capacity that can be resized without dropping work."""
    def __init__(self,limit):
        self.limit=limit;self.active=0;self.waiting=0
        self.condition=asyncio.Condition()

    async def __aenter__(self):
        self.waiting+=1
        try:
            async with self.condition:
                await self.condition.wait_for(lambda:self.active<self.limit)
                self.active+=1
        finally:self.waiting-=1
        return self

    async def __aexit__(self,*exception):
        async with self.condition:
            self.active-=1;self.condition.notify(1)

    async def resize(self,limit):
        if limit==self.limit:return
        async with self.condition:
            self.limit=limit;self.condition.notify_all()

    def metrics(self):
        return {'capacity':self.limit,'active':self.active,'waiting':self.waiting}


class NativeClient:
    def __init__(self,session,servers,concurrency=16,sampling=None,template_kwargs=None,request_timeout=600,transport_attempts=3):
        self.session,self.servers=session,servers
        self.sampling=sampling or {};self.template_kwargs=template_kwargs
        self.request_timeout=request_timeout
        self.transport_attempts=transport_attempts
        self.slots=[RequestGate(concurrency) for _ in servers]
        self.index=0
        self.eligible=list(range(len(servers)))

    def holdout(self,devices):
        reserved={(row['ip'],str(row['gpu'])) for row in devices}
        eligible=[i for i,s in enumerate(self.servers) if (s['ip'],str(s['gpus'][0])) not in reserved]
        assert eligible,'At least one production server is required'
        self.eligible=eligible

    def metrics(self):
        return {key:sum(self.slots[i].metrics()[key] for i in self.eligible) for key in ['capacity','active','waiting']}

    async def chat(self,messages,*,thinking=False,tools=None,choice=None,budget=2048,seed=0,response_format=None,sampling_overrides=None):
        index=self.eligible[self.index%len(self.eligible)];self.index+=1
        server=self.servers[index]
        payload={'model':server['model'],'messages':messages,'max_tokens':budget,'stream':False,
            'temperature':1. if thinking else .7,'top_p':.95 if thinking else .8,'top_k':20,'seed':seed}
        if server['model']=='Qwen/Qwen3.8-27B':
            payload['chat_template_kwargs']={'enable_thinking':thinking,'preserve_thinking':True,'reasoning_effort':'medium' if thinking else 'low'}
            payload['presence_penalty']=0 if thinking else 1.5
        if tools:payload.update(tools=tools,tool_choice=choice or 'auto')
        if response_format:payload['response_format']=response_format
        payload.update(self.sampling)
        if sampling_overrides is not None:payload.update(sampling_overrides)
        if self.template_kwargs is not None:payload['chat_template_kwargs']=self.template_kwargs
        async with self.slots[index]:
            body=await asyncio.to_thread(native_body,payload)
            transport_errors=[]
            for attempt in range(self.transport_attempts):
                try:
                    async with self.session.post(server['url']+'/v1/chat/completions',data=body,headers={'Content-Type':'application/json'},
                        timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as response:
                        raw=await response.json()
                        if response.status!=200:
                            transport_errors.append({'attempt':attempt+1,'http_status':response.status,'response':raw})
                            if response.status==400 and raw.get('error',{}).get('message')=='Already borrowed' and attempt<self.transport_attempts-1:
                                await asyncio.sleep(.1*(attempt+1));continue
                            raise RuntimeError(f'Native API HTTP {response.status}: {str(raw)[:900]}')
                    message=raw['choices'][0]['message']
                    saved={k:v for k,v in payload.items() if k!='messages'};saved['messages']=json.loads(json.dumps(messages))
                    return {'request':saved,'response':raw,'message':message,'server':server['url'],
                            'finish_reason':raw['choices'][0].get('finish_reason'),'recorded_at':progress.now(),'transport_errors':transport_errors}
                except asyncio.TimeoutError as error:
                    raise asyncio.TimeoutError(f'Native API exceeded {self.request_timeout} seconds at {server["url"]}; deferred instead of repeating this request') from error
                except aiohttp.ClientConnectionError:
                    if attempt==self.transport_attempts-1:raise
                    await asyncio.sleep(2+attempt)
