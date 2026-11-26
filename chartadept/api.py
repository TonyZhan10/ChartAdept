"""Send JSON requests to the configured model endpoint."""
import asyncio
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

DEFAULT_CONFIG = Path(__import__('os').environ.get('CHARTADEPT_API_CONFIG', 'api-config.json'))


def load_config(path=None):
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())


def endpoint(config, path):
    base = config['base_url'].rstrip('/')
    parts = urlsplit(base)
    prefix = parts.path.rstrip('/')
    if prefix and path.startswith(prefix + '/'):
        path = path[len(prefix):]
    return urlunsplit((parts.scheme, parts.netloc, prefix + '/' + path.lstrip('/'), '', ''))


def chat_payload(content, config=None, max_tokens=None):
    config = config or load_config()
    payload = {
        'model': config['model'],
        'messages': [
            {'role': 'system', 'content': 'Review the input and return a JSON object.'},
            {'role': 'user', 'content': content},
        ],
        'reasoning_effort': config.get('reasoning_effort', 'low'),
        'temperature': config.get('temperature', 0), 'stream': False,
        'max_tokens': max_tokens if max_tokens is not None else config.get('default_max_tokens', 4096),
        'response_format': {'type': 'json_object'},
    }
    if config.get('reasoning_effort', 'low') is None:
        payload.pop('reasoning_effort')
    if 'chat_template_kwargs' in config:
        payload['chat_template_kwargs'] = config['chat_template_kwargs']
    return payload


def request_json(method, path, payload, timeout=600, retries=0, *, config_path=None):
    config = load_config(config_path)
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    for attempt in range(retries + 1):
        try:
            request = Request(endpoint(config, path), data=body, method=method,
                              headers={'Authorization': 'Bearer ' + config['api_key'],
                                       'Content-Type': 'application/json'})
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as error:
            if attempt == retries:
                raise RuntimeError(f'GPT API HTTP {error.code}') from None
        except OSError:
            if attempt == retries:
                raise
        time.sleep(2 + attempt)


async def async_request_json(session, method, path, payload, timeout=600, retries=0, *, config=None):
    """Same request policy using the caller's async connection pool and limiter."""
    import aiohttp
    config = config or load_config()
    for attempt in range(retries + 1):
        try:
            async with session.request(method, endpoint(config, path), json=payload,
                    headers={'Authorization': 'Bearer ' + config['api_key']},
                    timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status != 200:
                    raise RuntimeError(f'Judge HTTP {response.status}: request failed')
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
            if attempt == retries:
                raise
        await asyncio.sleep(2 + attempt)
