"""Stream Sol reviews, retaining partial responses for interrupted requests."""
import asyncio
import json
import time

from aiohttp import ClientTimeout

from chartadept.api import endpoint
from chartadept import io as progress

TRANSPORT_REVISION = 'json-content-progress-idle-v1'


async def streamed_review(session, payload, config, output, timeout=600, idle_timeout=180):
    started=time.monotonic()
    raw={'choices':[{'message':{'content':''},'finish_reason':None}],'usage':{},
         'transport':{'stream':True,'first_event_seconds':None,'first_content_seconds':None,
                      'revision':TRANSPORT_REVISION}}
    transport=raw['transport']
    failure=None
    last_progress=None
    try:
        async with session.post(endpoint(config,'/v1/chat/completions'),json=payload,
            headers={'Authorization':'Bearer '+config['api_key']},
            timeout=ClientTimeout(total=timeout,sock_connect=30,sock_read=idle_timeout)) as response:
            transport.update(http_status=response.status,response_headers_seconds=time.monotonic()-started)
            if response.status!=200:
                raise RuntimeError(f'Judge HTTP {response.status}: request failed')
            if response.content_type=='application/json':
                # Some compatible gateways return a complete JSON response.
                completed=await response.json()
                completed['transport']=transport
                raw=completed
            else:
                async for line in response.content:
                    line=line.decode().strip()
                    if not line.startswith('data:'):
                        continue
                    if transport['first_event_seconds'] is None:
                        transport['first_event_seconds']=time.monotonic()-started
                    value=line[5:].strip()
                    if value=='[DONE]':
                        break
                    event=json.loads(value)
                    for key in ['id','model']:
                        if key in event:raw[key]=event[key]
                    if event.get('usage'):raw['usage']=event['usage']
                    for choice in event.get('choices',[]):
                        content=choice.get('delta',{}).get('content') or ''
                        if content:
                            if last_progress is None or content.strip():
                                last_progress=time.monotonic()
                            if transport['first_content_seconds'] is None:
                                transport['first_content_seconds']=time.monotonic()-started
                            raw['choices'][0]['message']['content']+=content
                        if choice.get('finish_reason'):
                            raw['choices'][0]['finish_reason']=choice['finish_reason']
                    if last_progress is not None and time.monotonic()-last_progress>=idle_timeout:
                        # A compatible gateway can keep an incomplete JSON stream
                        # alive with spaces. Reuse the existing idle deadline for
                        # content progress; preserve a complete usable object.
                        text=raw['choices'][0]['message']['content']
                        try:
                            complete=isinstance(json.loads(text),dict)
                        except json.JSONDecodeError:
                            complete=False
                        if complete:
                            transport['complete_json_before_whitespace_stall']=True
                            return raw
                        raise asyncio.TimeoutError(
                            f'Sol JSON content made no non-whitespace progress for {idle_timeout}s; reward remains pending')
            return raw
    except BaseException as error:
        failure=f'{type(error).__name__}: {error}'
        raise
    finally:
        transport['seconds']=time.monotonic()-started
        if failure:transport['error']=failure
        progress.write_json(output,raw)
