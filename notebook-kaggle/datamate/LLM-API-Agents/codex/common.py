"""Helpers for direct Codex calculations on released DataMate tables; no model calls."""
from pathlib import Path
import os,re,json,math,unicodedata,hashlib,datetime,pickle
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'dataset/datamate').exists())
DATA=Path(os.environ.get('DATAMATE_DATA',ROOT/'dataset/datamate/public_test'))
OUT=Path(__file__).resolve().parent
CACHE=Path('/tmp/datamate-codex-cache');CACHE.mkdir(exist_ok=True)
QUESTIONS=[json.loads(x) for x in (DATA/'questions.jsonl').read_text().splitlines()]
def text(v):
 if v is None or (not isinstance(v,(list,dict,tuple)) and pd.isna(v)):return ''
 return str(v).strip()
def norm(v):
 s=unicodedata.normalize('NFD',text(v).lower()).replace('đ','d')
 s=''.join(c for c in s if unicodedata.category(c)!='Mn')
 return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9]+',' ',s)).strip()
def num(v):
 if not text(v):return 0.
 if isinstance(v,(int,float)):return float(v)
 s=text(v).lower().replace('−','-').replace(' ','')
 s=s.replace('triệu','tr').replace('trieu','tr').replace('nghìn','k').replace('nghin','k')
 m=re.match(r'^(-?\d+(?:[.,]\d+)?)tr(\d*)',s)
 if m:return round(float(m[1].replace(',','.'))*1e6+(int(m[2])*1000 if m[2] else 0),6)
 m=re.match(r'^(-?\d+(?:[.,]\d+)?)k',s)
 if m:return round(float(m[1].replace(',','.'))*1000,6)
 s=re.sub(r'(?<=[0-9])m[23]$','',s)
 s=re.sub(r'[^0-9,.\-]','',s)
 if not s or s=='-':return 0.
 if ',' in s and '.' in s:
  # CSV money: 1,090,000.0000; localized 1.234,50 supported.
  if s.rfind('.')>s.rfind(','):s=s.replace(',','')
  else:s=s.replace('.','').replace(',','.')
 elif ',' in s:
  if s.count(',')>1 or len(s.rsplit(',',1)[1])==3:s=s.replace(',','')
  else:s=s.replace(',','.')
 elif s.count('.')>1 or ('.' in s and len(s.rsplit('.',1)[1])==3):s=s.replace('.','')
 return float(s)
def rnd(v,places=0):
 d=Decimal(str(float(v))).quantize(Decimal(1).scaleb(-places),rounding=ROUND_HALF_UP)
 return int(d) if places==0 else float(d)
def dt(v,year=None):
 if not text(v):return pd.NaT
 if isinstance(v,(pd.Timestamp,datetime.datetime,datetime.date)):return pd.Timestamp(v)
 s=text(v)
 if re.fullmatch(r'\d{5}(\.0)?',s):return pd.Timestamp('1899-12-30')+pd.Timedelta(days=float(s))
 s=re.sub(r'(\d+)\s*thg\s*(\d+)\s*(\d{4})',r'\1/\2/\3',s)
 if re.fullmatch(r'\d{1,2}/\d{1,2}',s):s+=f'/{year or 2025}'
 try:return pd.to_datetime(s,dayfirst=not bool(re.match(r'^\d{4}-',s)),format='mixed')
 except (ValueError,TypeError):return pd.NaT
def book(p):
 p=Path(p);key=hashlib.sha256((str(p.resolve())+str(p.stat().st_mtime_ns)).encode()).hexdigest()[:20]
 cache=CACHE/(key+'.pkl')
 if cache.exists():return pd.read_pickle(cache)
 x=pd.read_excel(p,sheet_name=None,header=None,engine='calamine');pd.to_pickle(x,cache);return x
def tab(p,sheet=0,header=None):
 b=book(p);raw=b[list(b)[sheet] if isinstance(sheet,int) else sheet]
 if header is None:
  header=max(range(min(12,len(raw))),key=lambda i: raw.iloc[i].notna().sum())
 df=raw.iloc[header+1:].copy();df.columns=[text(v) or f'_blank{i}' for i,v in enumerate(raw.iloc[header])]
 return df.reset_index(drop=True)
def series(df,col,kind='text'):
 s=df[col] if col in df else pd.Series('',index=df.index)
 return s.map(num if kind=='num' else norm if kind=='norm' else text)
def output(typ,answer=None,reason=None):
 r={'type':typ}
 if typ not in ('unanswerable','abstain'):r['answer']=answer
 if reason:r['reason']=reason
 return r
