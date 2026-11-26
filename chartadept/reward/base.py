import asyncio,os
from aiohttp import web
class Broker:
    async def score(self,request):
        payload=await request.json()
        trace=payload['metadata']['chart_trace_file']
        async with self.trace_locks.setdefault(trace,asyncio.Lock()):
            return await self.score_payload(payload)
    async def health(self,request):
        return web.json_response({'status':'running','pid':os.getpid(),
            'judge_model':self.judge_model,**dict(self.counts)})
