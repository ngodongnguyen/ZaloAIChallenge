"""Dual ASR prompt data and paired evaluation. No labels enter transcript assembly."""
from __future__ import annotations

import json
import math
import random
import unicodedata
from pathlib import Path


def combine_transcripts(primary, secondary, needed):
    """Retain two recognitions of the same audio; preserve exact primary on agreement.

    Both final and history audio are required. This does not decide which ASR is
    correct, concatenate two user intents, or use ground-truth transcripts.
    """
    combined = {}
    for audio in sorted(set(needed)):
        pair = [primary.get(audio), secondary.get(audio)]
        if any(not isinstance(text, str) or not text.strip() for text in pair):
            raise ValueError(f'Missing/empty ASR transcript for {audio}')
        a, b = pair
        normalize = lambda text: ' '.join(unicodedata.normalize('NFC', text).casefold().split())
        if normalize(a) == normalize(b):
            combined[audio] = a
        else:
            combined[audio] = (
                'Hai bản nhận dạng của CÙNG MỘT lời nói (không phải hai yêu cầu):\n'
                + 'Whisper: ' + json.dumps(a, ensure_ascii=False) + '\n'
                + 'Gipformer: ' + json.dumps(b, ensure_ascii=False) + '\n'
                + 'Đối chiếu theo ngữ cảnh hội thoại và kết quả tool để hiểu đúng một yêu cầu. '
                  'Ưu tiên Whisper khi chưa có căn cứ rõ để sửa; không ghép hai bản thành các việc khác nhau '
                  'và không thực hiện lặp lại. Nếu thực sự thiếu thông tin bắt buộc, dùng quy tắc hỏi lại.'
            )
    return combined


def score_table(records, expected_ids=None):
    table = {}
    for record in records:
        qid, score = record.get('id'), record.get('score')
        if (not isinstance(qid, str) or qid in table or record.get('pending')
                or isinstance(score, bool) or not isinstance(score, (float, int))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Incomplete, duplicate or invalid per-ID scores')
        table[qid] = record
    if not table or (expected_ids is not None and set(table) != set(expected_ids)):
        raise ValueError('Score IDs differ from the frozen dev set')
    return table


def checked_control(directory, ids, checkpoint_sha, asr_sha):
    directory = Path(directory)
    cfg = json.loads((directory/'config.json').read_text())
    if (cfg.get('checkpoint_sha256') != checkpoint_sha or cfg.get('asr_source_sha256') != asr_sha
            or not isinstance(cfg.get('subset'), list) or set(cfg['subset']) != set(ids)):
        raise ValueError('Control checkpoint, ASR cache or dev IDs differ')
    records = [json.loads(line) for line in (directory/'per_id_scores.jsonl').read_text().splitlines() if line.strip()]
    return score_table(records, ids)


def paired_report(candidate, control):
    candidate = score_table([dict(r, id=qid) for qid, r in candidate.items()])
    control = score_table([dict(r, id=qid) for qid, r in control.items()], candidate)
    ids = sorted(candidate)
    deltas = [candidate[qid]['score']-control[qid]['score'] for qid in ids]
    n = len(deltas)
    rng = random.Random(2026)
    bootstrap = sorted(100*sum(rng.choices(deltas, k=n))/n for _ in range(2000))
    return {'n': n, 'candidate_score': 100*sum(r['score'] for r in candidate.values())/n,
            'control_score': 100*sum(r['score'] for r in control.values())/n,
            'delta_points': 100*sum(deltas)/n, 'ci95': [bootstrap[49],bootstrap[1949]],
            'improved': sum(d>1e-9 for d in deltas), 'worsened': sum(d<-1e-9 for d in deltas),
            'unchanged': sum(abs(d)<=1e-9 for d in deltas)}
