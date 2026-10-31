"""Controlled 30-run voice latency and shared-memory comparison on the 395."""
import asyncio
import base64
import io
import json
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path

import aiohttp
import soundfile as sf
import bench_e2e as bench

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'comparison-20261016'
OUT.mkdir(exist_ok=True)
MODELS = {
    'qwen3.6': Path.home() / 'qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf',
    'nemotron-bolt': Path.home() / 'nemotron-3-5-bolt-74b-a9b-24gb/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf',
}
BIN = Path.home() / 'llama.cpp/build/bin/llama-server'
FLAGS = ['--host', '127.0.0.1', '--port', '9931', '-ngl', '99', '-c', '8192', '-fit', 'off', '--jinja', '--reasoning', 'off', '--parallel', '1']
report = {'conditions': {'runs_per_path': 30, 'warmups_per_path': 2, 'audio': 'human_river.wav', 'reference': '日本で一番長い川は', 'flags': FLAGS, 'binary': str(BIN), 'cache': 'enabled, repeated identical input', 'temperature': 0.7, 'max_tokens': 512, 'speaker_id': 3, 'memory': 'MemTotal - MemAvailable, sampled every 0.2s; includes all processes; GPU memory overlaps RAM'}, 'models': {}}
current_capture = {}


def save():
    (OUT / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


def memory():
    m = {p[0].rstrip(':'): int(p[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(p := line.split()) >= 2}
    return {'time': time.time(), 'used_bytes': m['MemTotal'] - m['MemAvailable'], 'swap_used_bytes': m['SwapTotal'] - m['SwapFree']}


async def sampler(samples, done):
    while not done.is_set():
        samples.append(memory())
        await asyncio.sleep(0.2)


async def healthy():
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get('http://127.0.0.1:9931/health', timeout=aiohttp.ClientTimeout(total=2)) as r:
                return r.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False


async def stop_server():
    subprocess.run(['tmux', 'send-keys', '-t', 'aiassistant:llama', 'C-c'], check=True)
    for _ in range(100):
        p = subprocess.run(['pgrep', '-f', rf'^{re.escape(str(BIN))} .*--port 9931'], capture_output=True, text=True)
        if p.returncode == 1:
            await asyncio.sleep(1)
            return
        await asyncio.sleep(0.2)
    raise RuntimeError('Server did not stop')


async def start_server(label):
    command = ['env', 'ROCM_PATH=/opt/rocm', 'HIP_VISIBLE_DEVICES=0', 'LD_LIBRARY_PATH=/usr/local/lib:/opt/rocm/lib:/opt/rocm/lib/llvm/lib', str(BIN), '-m', str(MODELS[label]), *FLAGS]
    cmd = shlex.join(command) + ' > ' + shlex.quote(str(OUT / (label + '-server.log'))) + ' 2>&1'
    subprocess.run(['tmux', 'send-keys', '-t', 'aiassistant:llama', '-l', cmd], check=True)
    subprocess.run(['tmux', 'send-keys', '-t', 'aiassistant:llama', 'Enter'], check=True)
    for _ in range(600):
        if await healthy():
            async with aiohttp.ClientSession() as s:
                async with s.get('http://127.0.0.1:9931/v1/models') as r:
                    data = await r.json()
            assert data['data'][0]['id'] == str(MODELS[label]), data
            return
        await asyncio.sleep(1)
    raise RuntimeError('Server did not become healthy')


async def first_speak(ready):
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect('ws://localhost:8000/ws') as ws:
            ready.set()
            async with asyncio.timeout(180):
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(msg.data)
                    if event.get('type') == 'error':
                        raise RuntimeError(event)
                    if event.get('type') == 'speak':
                        stamp = time.perf_counter()
                        audio, sr = sf.read(io.BytesIO(base64.b64decode(event['audio'])))
                        current_capture.update(first_text=event.get('text'), first_audio_duration_sec=len(audio)/sr)
                        return stamp
    raise RuntimeError('No speech')


async def main():
    bench._first_speak = first_speak
    wav = ROOT / 'audio/human_river.wav'
    audio, sr = sf.read(wav, dtype='float32')
    assert sr == 16000
    webm = bench._to_webm(wav)
    report['conditions']['audio_duration_sec'] = len(audio)/sr
    async with aiohttp.ClientSession() as s:
        async with s.get('http://localhost:8001/health') as r:
            report['conditions']['health'] = await r.json()
    for label in MODELS:
        print('Preparing', label, flush=True)
        await stop_server()
        baseline_samples = []
        for _ in range(10):
            baseline_samples.append(memory())
            await asyncio.sleep(0.2)
        baseline = statistics.median(x['used_bytes'] for x in baseline_samples)
        row = report['models'][label] = {'model': str(MODELS[label]), 'baseline_used_bytes': baseline, 'runs': [], 'warmups': []}
        samples, done = [], asyncio.Event()
        sampling = asyncio.create_task(sampler(samples, done))
        try:
            await start_server(label)
            row['loaded_idle_used_bytes'] = memory()['used_bytes']
            for i in range(32):
                for path, fn in [('batch', lambda: bench._measure_batch(webm)), ('stream', lambda: bench._measure_stream(audio))]:
                    current_capture.clear()
                    latency = await fn()
                    assert latency is not None and latency > 0
                    item = {'path': path, 'index': i-1, 'latency_ms': latency, **current_capture}
                    row['warmups' if i < 2 else 'runs'].append(item)
                    save()
                if i >= 2:
                    print(label, f'{i-1}/30', 'batch/stream', *(round(x['latency_ms'], 1) for x in row['runs'][-2:]), flush=True)
            for path in ['batch', 'stream']:
                row[path + '_ms'] = bench._stats([x['latency_ms'] for x in row['runs'] if x['path'] == path])
        finally:
            done.set()
            await sampling
            (OUT / (label + '-memory.json')).write_text(json.dumps(samples))
            row['peak_used_bytes'] = max(x['used_bytes'] for x in samples)
            row['peak_minus_baseline_bytes'] = row['peak_used_bytes'] - baseline
            row['peak_swap_used_bytes'] = max(x['swap_used_bytes'] for x in samples)
            save()
        print('Finished', label, json.dumps({k: v for k, v in row.items() if k.endswith('_ms') or k.endswith('_bytes')}), flush=True)
    print('DONE; Nemotron remains running.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
