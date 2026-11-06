"""New ASR calibration using official train text, plus a public diagnostic probe.

Only Deepgram is called. Credentials are prompted and never persisted.
"""
import argparse
import concurrent.futures
import getpass
import hashlib
import io
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'dataset/lifemate/train+public_test'
OUT = HERE / 'results/codex_v2'


def tokens(text):
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).lower())


def distance(a, b):
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(x != y)))
        prev = cur
    return prev[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public-all', action='store_true')
    parser.add_argument('--public-probe', action='store_true')
    parser.add_argument('--hints', action='store_true')
    parser.add_argument('--nova2', action='store_true')
    parser.add_argument('--speed',type=float,choices=[0.9,1.1])
    parser.add_argument('--public-ids',nargs='+')
    parser.add_argument('--probe-model',default='nova-2',choices=['nova-2','nova-3-speed-0.9','whisper-large'])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    public = [json.loads(l) for l in (ROOT/'public_test/public_test.jsonl').read_text().splitlines()]
    if args.public_ids:
        requested = set(args.public_ids)
        selected = [r for r in public if r['id'] in requested]
        assert {r['id'] for r in selected} == requested
        jobs = [('public_test',r['audio'],args.probe_model,None) for r in selected]
    elif args.hints or args.nova2 or args.speed:
        variant = f'nova-3-speed-{args.speed}' if args.speed else ('nova-2' if args.nova2 else 'nova-3-keyterm')
        train = [json.loads(l) for l in (ROOT/'train/train.jsonl').read_text().splitlines()]
        selected = [train[i] for i in [0,1,2,3,4,5,150,300,450,600,750,900,1050,1200,1350,1500]]
        jobs = [('train', r['audio'], variant, r['audio_text']) for r in selected]
        probe = {'public_0003','public_0004','public_0027','public_0510','public_0653','public_0196','public_0426'}
        jobs += [('public_test', r['audio'], variant, None) for r in public if r['id'] in probe]
    elif args.public_all:
        audios = sorted({r['audio'] for r in public} | {h['audio'] for r in public for h in r['history'] if h.get('audio')})
        jobs = [('public_test', a, 'whisper-large', None) for a in audios]
    elif args.public_probe:
        probe = {'public_0003','public_0004','public_0027','public_0510','public_0653','public_0196','public_0426'}
        jobs = [('public_test', r['audio'], 'whisper-large', None) for r in public if r['id'] in probe]
    else:
        train = [json.loads(l) for l in (ROOT/'train/train.jsonl').read_text().splitlines()]
        selected = [train[i] for i in [0,1,2,3,4,5,150,300,450,600,750,900,1050,1200,1350,1500]]
        jobs = [('train', r['audio'], m, r['audio_text']) for r in selected for m in ['nova-3', 'whisper-large']]
        probe = {'public_0003','public_0004','public_0027','public_0510','public_0653','public_0196','public_0426'}
        jobs += [('public_test', r['audio'], 'whisper-large', None) for r in public if r['id'] in probe]
    jobs = [j for j in jobs if not (OUT/'asr'/j[2]/(Path(j[1]).stem+'.json')).exists()]
    if jobs:
        key = getpass.getpass('Deepgram API key (not saved): ')

        def request(job):
            split, relative, model, reference = job
            data = (ROOT/split/relative).read_bytes()
            original_hash = hashlib.sha256(data).hexdigest()
            if model.startswith('nova-3-speed-'):
                speed = float(model.rsplit('-',1)[-1])
                with wave.open(io.BytesIO(data)) as source:
                    channels,width,rate,frames,_,_ = source.getparams()
                    samples = source.readframes(frames)
                buffer = io.BytesIO()
                with wave.open(buffer,'wb') as target:
                    target.setnchannels(channels)
                    target.setsampwidth(width)
                    target.setframerate(round(rate*speed))
                    target.writeframes(samples)
                data = buffer.getvalue()
            params = {'model': 'nova-3' if model.startswith('nova-3-') else model,'language':'vi','punctuate':'true','smart_format':'false'}
            if model == 'nova-3-keyterm':
                params['keyterm'] = ['Sếp Dũng','Minh','Lan','Tuấn','Cô giáo Mai','Bác Ba','đặt báo thức','trước giờ bay','chuyến bay','cất cánh','hạ cánh','nửa tiếng','đá bóng','chạy bộ','đèn phòng khách','giá vàng SJC','chuyển ví','nhắn SMS','nhắn Zalo','nạp tiền điện thoại','mồng mười','tháng mười hai']
            query = urllib.parse.urlencode(params, doseq=True)
            for attempt in range(5):
                try:
                    req = urllib.request.Request('https://api.deepgram.com/v1/listen?'+query, data=data, headers={'Authorization':'Token '+key,'Content-Type':'audio/wav'})
                    with urllib.request.urlopen(req, timeout=180) as response:
                        raw = json.load(response)
                    alt = raw['results']['channels'][0]['alternatives'][0]
                    result = {'split':split,'audio':relative,'audio_sha256':original_hash,'request_audio_sha256':hashlib.sha256(data).hexdigest(),'request_parameters':params,'transcript':alt['transcript'],'confidence':alt.get('confidence'),'metadata':raw.get('metadata'),'reference':reference}
                    path = OUT/'asr'/model/(Path(relative).stem+'.json')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.with_suffix('.tmp').write_text(json.dumps(result,ensure_ascii=False))
                    path.with_suffix('.tmp').replace(path)
                    return relative, model, None
                except urllib.error.HTTPError as e:
                    if e.code not in (429,500,502,503,504):
                        return relative,model,f'HTTP {e.code}: '+e.read(350).decode(errors='replace')
                except (OSError,TimeoutError) as e:
                    if attempt == 4:
                        return relative,model,str(e)
                time.sleep(2**attempt)
            return relative,model,'retry limit'

        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for n, fut in enumerate(concurrent.futures.as_completed([pool.submit(request,j) for j in jobs]),1):
                result = fut.result()
                if result[-1]:
                    failures.append(result)
                    print(json.dumps(result,ensure_ascii=False),flush=True)
                if n % 25 == 0 or n == len(jobs):
                    print(f'{n}/{len(jobs)} requests; failures={len(failures)}',flush=True)
        if failures:
            raise SystemExit(1)
    metrics = {}
    for model in ['nova-3','whisper-large','nova-3-keyterm','nova-2','nova-3-speed-0.9','nova-3-speed-1.1']:
        total = errors = count = 0
        for path in (OUT/'asr'/model).glob('train_*.json'):
            row = json.loads(path.read_text())
            ref = tokens(row['reference'])
            errors += distance(ref,tokens(row['transcript']))
            total += len(ref)
            count += 1
        if count:
            metrics[model] = {'samples':count,'word_errors':errors,'reference_words':total,'wer':errors/total}
    (OUT/'asr_calibration.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
