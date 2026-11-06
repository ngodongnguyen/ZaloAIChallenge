"""Deepgram ASR of official public inputs only; key is prompted, never persisted."""
import argparse
import concurrent.futures
import getpass
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (args.root / 'public_test.jsonl').read_text().splitlines() if x.strip()]
    paths = sorted({r['audio'] for r in rows} | {h['audio'] for r in rows for h in r['history'] if h.get('audio')})
    paths = [p for p in paths if not (args.output / (Path(p).stem + '.json')).exists()]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print('All audio already transcribed.', flush=True)
        return
    key = getpass.getpass('Deepgram API key (not saved): ')
    url = 'https://api.deepgram.com/v1/listen?model=nova-3&language=vi&smart_format=true&punctuate=true'

    def transcribe(relative):
        data = (args.root / relative).read_bytes()
        for attempt in range(5):
            request = urllib.request.Request(url, data=data, headers={'Authorization': 'Token ' + key, 'Content-Type': 'audio/wav'})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                alternative = result['results']['channels'][0]['alternatives'][0]
                result = {'audio': relative, 'audio_sha256': hashlib.sha256(data).hexdigest(), 'transcript': alternative['transcript'], 'confidence': alternative.get('confidence'), 'metadata': result.get('metadata'), 'words': alternative.get('words', [])}
                target = args.output / (Path(relative).stem + '.json')
                target.with_suffix('.tmp').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
                target.with_suffix('.tmp').replace(target)
                return relative, None
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    return relative, f'HTTP {exc.code}: {exc.read(600).decode(errors="replace")}'
            except (OSError, TimeoutError) as exc:
                if attempt == 4:
                    return relative, type(exc).__name__
            time.sleep(2 ** attempt)
        return relative, 'retries exhausted'

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, future in enumerate(concurrent.futures.as_completed([pool.submit(transcribe, p) for p in paths]), 1):
            path, error = future.result()
            if error:
                failures.append({'audio': path, 'error': error})
                print(json.dumps(failures[-1], ensure_ascii=False), flush=True)
            if n % 50 == 0 or n == len(paths):
                print(f'Completed {n}/{len(paths)}; failures={len(failures)}', flush=True)
    (args.output.parent / 'asr_failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
