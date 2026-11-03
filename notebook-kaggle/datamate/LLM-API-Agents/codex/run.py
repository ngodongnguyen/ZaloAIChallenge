#!/usr/bin/env python3
"""Recompute the direct Codex analysis, validate it, and export submission.jsonl."""
import argparse
import collections
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_QUESTIONS_SHA256 = 'b33ae585ed5d4acdacbd6a2873d442e3a5673a8238b6e932827f40c48c158832'
EXPECTED_RULEBOOK_SHA256 = '959ebb63b709fad68f7e8a339cde955c96b6e9af2e48141db3ad5a2429510e10'
GROUPS = {
 'root': {'dev_001','dev_002','dev_003','dev_004','dev_009','dev_010'},
 'pharmacy_pos': {'dev_005','dev_006'},
 'retail_baby': {'dev_007','dev_008','dev_013'},
 'phone_repair': {'dev_011','dev_012','dev_014'},
}

def digest(path):
 h = hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def numeric(v):
 return type(v) in (int,float) and math.isfinite(v)

def scalar(v):
 return isinstance(v,str) or numeric(v)

def validate_row(r):
 assert isinstance(r,dict),r
 assert isinstance(r.get('id'),str) and r['id'],r
 t=r.get('type')
 assert t in {'number','percent','date','string','boolean','list','table','unanswerable','abstain'},r
 allowed={'id','type','reason'} if t in {'unanswerable','abstain'} else {'id','type','answer'}
 assert set(r)<=allowed,r
 if 'reason' in r:assert isinstance(r['reason'],str),r
 if t=='number':
  assert numeric(r.get('answer')),r
  if isinstance(r['answer'],float) and r['answer'].is_integer():
   r['answer']=int(r['answer'])
 elif t=='percent':
  assert numeric(r.get('answer')),r
 elif t=='string':assert isinstance(r.get('answer'),str) and r['answer'].strip(),r
 elif t=='boolean':assert type(r.get('answer')) is bool,r
 elif t=='date':
  a=r.get('answer');assert isinstance(a,str) and re.fullmatch(r'\d{4}-\d{2}(-\d{2})?',a),r
  datetime.strptime(a,'%Y-%m-%d' if len(a)==10 else '%Y-%m')
 elif t=='list':
  assert isinstance(r.get('answer'),list) and all(scalar(v) for v in r['answer']),r
  r['answer']=[int(v) if isinstance(v,float) and v.is_integer() else v for v in r['answer']]
 elif t=='table':
  a=r.get('answer');assert isinstance(a,dict) and set(a)=={'columns','rows'},r
  assert isinstance(a['columns'],list) and a['columns'] and all(isinstance(c,str) and c for c in a['columns']),r
  assert isinstance(a['rows'],list),r
  assert all(isinstance(row,list) and len(row)==len(a['columns']) and all(scalar(v) for v in row) for row in a['rows']),r
  a['rows']=[[int(v) if isinstance(v,float) and v.is_integer() else v for v in row] for row in a['rows']]
 json.dumps(r,ensure_ascii=False,allow_nan=False)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--merge-only',action='store_true',help='Validate and combine already computed answer maps.')
 args=parser.parse_args()
 from common import DATA,QUESTIONS
 started=time.time()
 assert digest(DATA/'questions.jsonl')==EXPECTED_QUESTIONS_SHA256,'Question list changed; these public-test query plans need review.'
 assert digest(DATA/'rulebook.md')==EXPECTED_RULEBOOK_SHA256,'Rulebook changed; re-audit query plans first.'
 if not args.merge_only:
  for name in GROUPS:
   subprocess.run([sys.executable,'-u',str(HERE/f'solve_{name}.py')],check=True,cwd=HERE)
 from audit_root import audit
 reconciliations=audit()
 merged={}
 group_qa={}
 by_id={q['id']:q for q in QUESTIONS}
 assert len(by_id)==len(QUESTIONS)==1000
 for name,businesses in GROUPS.items():
  path=HERE/f'answers_{name}.json';answers=json.loads(path.read_text(encoding='utf-8'))
  expected={q['id'] for q in QUESTIONS if q['business_id'] in businesses}
  assert set(answers)==expected,(name,'missing',expected-set(answers),'extra',set(answers)-expected)
  assert not (set(merged)&set(answers)),name
  for key,r in answers.items():
   r={'id':key,**r}
   assert key==r['id'],(key,r)
   validate_row(r)
   answers[key]=r
  merged.update(answers)
  evidence_file=HERE/f'evidence_{name}.json'
  if evidence_file.exists():
   qa=json.loads(evidence_file.read_text(encoding='utf-8')).get('__qa__')
   if qa is not None:group_qa[name]=qa
 assert set(merged)==set(by_id)
 rows=[merged[q['id']] for q in QUESTIONS]
 target=HERE/'submission.jsonl';temporary=HERE/'submission.tmp.jsonl'
 temporary.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
 reloaded=[json.loads(line) for line in temporary.read_text(encoding='utf-8').splitlines()]
 assert reloaded==rows
 temporary.replace(target)
 report={
  'run_id':'dm-A-005-codex-public',
  'method':'Direct Codex reasoning and executed deterministic Python calculations on released data.',
  'question_count':len(rows),
  'types':dict(collections.Counter(r['type'] for r in rows)),
  'by_business':{b:dict(collections.Counter(merged[q['id']]['type'] for q in QUESTIONS if q['business_id']==b)) for b in sorted({q['business_id'] for q in QUESTIONS})},
  'checks':{'jsonl_and_answer_types':True,'all_public_ids_exactly_once':True,'question_order_preserved':True,'question_and_rulebook_version':True},
  'source_reconciliations':reconciliations,
  'group_source_reconciliations':group_qa,
  'question_sha256':digest(DATA/'questions.jsonl'),
  'rulebook_sha256':digest(DATA/'rulebook.md'),
  'submission_sha256':digest(target),
  'source_scripts':{p.name:digest(p) for p in sorted(HERE.glob('*.py'))},
  'elapsed_seconds_this_command':round(time.time()-started,3),
  'score':None,
  'score_note':'Public-test answers are hidden; schema and reconciliation checks do not measure answer accuracy. Not submitted to leaderboard.',
 }
 (HERE/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'submission':str(target),'rows':len(rows),'types':report['types'],'sha256':report['submission_sha256']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
