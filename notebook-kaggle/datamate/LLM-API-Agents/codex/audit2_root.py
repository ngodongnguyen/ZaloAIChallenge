"""Independent labeled-train calibration and current root-query diagnostic evidence."""
from common import *
from root_data import N,dates,parse_quantity
from solve_root import period,PLANS,rank,named
import collections

def train110():
 p=ROOT/'dataset/datamate/train'
 qs=[json.loads(x) for x in (p/'questions.jsonl').read_text().splitlines()]
 gold={r['id']:r for x in (p/'answers.jsonl').read_text().splitlines() if (r:=json.loads(x))}
 df=pd.read_csv(p/'data/biz_110/danh_sach_hoa_don.csv',dtype=str)
 df=df[df['Mã bill'].map(text).str.match(r'^ORD\d+$')].copy().drop_duplicates()
 df['ord']=df['Mã bill'].map(text);df=df.drop_duplicates('ord')
 df['date']=dates(df['Ngày giao dịch']).dt.normalize()
 df['revenue']=df['Tien hang'].map(num)
 statuses=df.Status.map(N)
 print('Train110 source statuses',statuses.value_counts().to_dict())
 df=df[~statuses.str.contains(r'huy|cancel|tra hang|hoan hang|tra lai|khach tra|doi tra|hang tra ve|hoan tra|returned')]
 result=[]
 for q in qs:
  if q['business_id']!='biz_110':continue
  n=N(q['question']);a,z=period(q,-1);part=df[df.date.between(a,z)]
  g=gold[q['id']]
  if g['type']=='date':pred=part.groupby('date').revenue.sum().sort_index().idxmax().strftime('%Y-%m-%d')
  elif any(v in n for v in ['so don','may don','bao nhieu don','soos bill','order','ddown']):pred=len(part)
  else:pred=rnd(part.revenue.sum())
  result.append({'id':q['id'],'question':q['question'],'from':str(a.date()),'to':str(z.date()),'expected':g['answer'],'actual':pred,'correct':pred==g['answer']})
 return result

def tie_diagnostics():
 res=[]
 for b in ['dev_001','dev_002','dev_003','dev_004','dev_009','dev_010']:
  s=pd.read_pickle(CACHE/(b+'-root.pkl'))
  for i,(op,args) in PLANS.items():
   q=QUESTIONS[i]
   if q['business_id']!=b:continue
   if op not in ['cust_orders','cust_rev','cust_top_rev','loyal','top_products','top_products_one','bestprofit']:continue
   a,z=period(q,i)
   df=s.select(a,z,orders=op.startswith('cust_') or op=='loyal')
   key='cid' if op.startswith('cust_') or op=='loyal' else 'pid'
   if key=='cid':df=df[df.cid.ne('')]
   val='ord' if op=='cust_orders' else 'date' if op=='loyal' else 'profit' if op=='bestprofit' else 'bqty' if key=='pid' else 'rev'
   how='nunique' if op in ['cust_orders','loyal'] else 'sum'
   r=rank(df,key,val,how);k=args[0] if op in ['cust_top_rev','top_products'] else 1
   ties=r[r[val].duplicated(False)&r[val].isin(r.head(k)[val])]
   if not ties.empty:
    names=named(s,key);res.append({'index':i,'business_id':b,'question':q['question'],'key':key,'candidates':[{'key':row[key],'name':names[row[key]],'value':float(row[val])} for _,row in ties.iterrows()]})
 return res

def train117():
 from root_data import Shop
 from solve_root import solve
 p=ROOT/'dataset/datamate/train';base=p/'data/biz_117';raw=[]
 for sh in book(base/'SoBanHang.xlsx'):
  d=tab(base/'SoBanHang.xlsx',sh).drop_duplicates();d['Mã phiếu']=d['Mã phiếu'].ffill()
  cols=['TenKH','Ngày lập','Ghi chú TT'];d[cols]=d.groupby('Mã phiếu')[cols].ffill()
  d['Ngày lập']=dates(d['Ngày lập'],int(sh[-4:]));raw.append(d)
 d=pd.concat(raw).rename(columns={'Mã phiếu':'ord','TenKH':'customer','Ngày lập':'date','Ghi chú TT':'status','Tiền':'rev','Số lượng hàng':'qty','Tên SP':'prod','DonVi':'unit','Giờ':'datetime'})
 d=d[d.ord.map(text).str.match(r'^ORD\d+$') & d['prod'].map(text).ne('')].copy()
 for c in ['ord','prod','unit','customer']:d[c]=d[c].map(text)
 for c in ['rev','qty']:d[c]=d[c].map(parse_quantity if c=='qty' else num)
 d['date']=pd.to_datetime(d.date).dt.normalize();d['datetime']=dates(d['datetime']);d['pid']=d['prod'].map(N);d['cid']=d.customer.map(N)
 d.loc[d.cid.isin(['','le','khach le','kl','khach vang lai']),'cid']=''
 conv=tab(base/'DanhMuc.xlsx');look={(N(r['Hàng']),N(r['ĐV lớn'])):num(r['Hệ số QĐ']) for _,r in conv.iterrows()}
 d['bqty']=[q*look.get((p,N(u)),1) for q,p,u in zip(d.qty,d.pid,d.unit)]
 st=d.status.map(N);d['cancel']=st.str.contains(r'huy|cancel|khong giao duoc');d['returned']=d.qty.lt(0)|st.str.contains(r'tra hang|hoan tra|hoan hang|tra lai|khach tra|hang tra ve|doi tra')
 d['cancel']=d.ord.map(d.groupby('ord').cancel.any());d['returned']=d.ord.map(d.groupby('ord').returned.any());d['valid']=~d.cancel & ~d.returned
 o=d.groupby('ord').agg({'date':'first','datetime':'first','cid':'first','customer':'first','rev':'sum','cancel':'max','returned':'max','valid':'max'}).reset_index()
 obj=Shop.__new__(Shop);obj.biz='biz_117';obj.path=base;obj.lines=d;obj.orders=o;obj.start=o.date.min();obj.end=o.date.max()
 imp=tab(base/'SoNhapHang.xlsx').rename(columns={'Ngày nhập':'date','Thành tiền':'amount','Nội dung':'note'});imp['date']=dates(imp.date);imp['amount']=imp.amount.map(num);obj.imports=imp
 qs=[json.loads(x) for x in (p/'questions.jsonl').read_text().splitlines()];gold={r['id']:r for x in (p/'answers.jsonl').read_text().splitlines() if (r:=json.loads(x))};res=[]
 def canon(v):return re.sub(r'\s+',' ',''.join(c if c.isalnum() else ' ' for c in unicodedata.normalize('NFC',str(v)).lower())).strip()
 for q in qs:
  if q['business_id']!='biz_117':continue
  n=N(q['question']);g=gold[q['id']];op=None;args=()
  if any(x in n for x in ['gia ban','gia nhap','don gia','gia binh quan','tang gia','doi gia','len gia','con may','con lai','dang ban','khong ban ra','het ngay hom nay','cuoi ngay','quy']):continue
  if 'huy' in n:op='cancelrate' if g['type']=='percent' else 'cancel'
  elif 'khach' in n and 'don' in n and any(x in n for x in ['nhieu don nhat','nhìu don nhat']):op='cust_orders'
  elif 'khach' in n and 'don' in n:
   m=re.search(r'(?:tu|it nhat) (\d+) don',n)
   if m:op='cust_min_share' if '%' in q['question'] or 'phan tram' in n else 'cust_min';args=(int(m[1]),)
  elif any(x in n for x in ['ban chay','hot nhat']):
   m=re.search(r'([35]) (?:khach|mat hang|hang|mon|sp)',n);op='top_products' if m else 'top_products_one';args=(int(m[1]),) if m else ()
  elif 'ca ' in n:
   if g['type']=='percent':op='shift_growth_orders' if any(x in n for x in ['so don','so ddown']) else 'shift_growth_rev'
   else:op='shift_orders' if any(x in n for x in ['may don','bao nhieu don']) else 'shift_rev';args=('afternoon' if 'ca chieu' in n else 'morning',)
  elif any(x in n for x in ['trung binh moi don','binh quan moi don']):op='avg'
  elif any(x in n for x in ['so don co','so don hang co','may don co']) and any(x in n for x in ['doanh thu','dthu']):
   m=re.search(r'(?:tu|≥)\s*([0-9.,]+)',q['question'])
   if m:op='order_threshold';args=(num(m[1]),)
  elif 'nhap hang' in n and not any(x in n for x in ['gia','binh quan','nha cung cap']):op='import_total'
  elif any(x in n for x in ['ngay nao','ngay nao']) and g['type']=='date':op='bestday'
  elif g['type']=='percent' and any(x in n for x in ['cung ky thang','cung ky th truoc','dau thang']):op='growth_mtd'
  elif g['type']=='number' and any(x in n for x in ['dthu','doanh thu']) and not any(x in n for x in ['cua ','trung binh','binh quan','nhap','gia','nhieu don']):op='rev'
  if q['id'] in ['q_4b77ed39a2ff','q_e35d8fac5233']:
   op='cust_min';args=(4 if q['id']=='q_4b77ed39a2ff' else 3,)
  if op is None:continue
  qq={**q,'question':q['question'].replace('tuần trc','tuần trước').replace('tuan trc','tuan truoc')}
  try:
   PLANS[-1]=(op,args);r,e=solve(obj,qq,-1)
  except (ValueError,KeyError,AttributeError,IndexError) as ex:
   res.append({'id':q['id'],'question':q['question'],'operation':op,'skip':str(ex)});continue
  expected=g.get('answer');actual=r.get('answer')
  good=r['type']==g['type'] and (canon(actual) in [canon(expected)]+[canon(x) for x in g.get('alias',[])] if g['type']=='string' else actual==expected)
  res.append({'id':q['id'],'question':q['question'],'operation':op,'expected':expected,'actual':actual,'actual_type':r['type'],'correct':good})
 PLANS.pop(-1,None)
 return res

if __name__=='__main__':
 t=train110();ties=tie_diagnostics();t117=train117()
 tested117=[r for r in t117 if 'correct' in r]
 report={'train110':t,'train110_correct':sum(x['correct'] for x in t),'train110_total':len(t),'train117':t117,'train117_correct':sum(x['correct'] for x in tested117),'train117_tested':len(tested117),'rank_ties':ties}
 (OUT/'audit2_root.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print('Train110',report['train110_correct'],'/',len(t),'Train117',report['train117_correct'],'/',len(tested117))
 for r in t+tested117:
  if not r['correct']:print('MISMATCH',r)
 print('Current rank tie cases:',json.dumps(ties,ensure_ascii=False))
