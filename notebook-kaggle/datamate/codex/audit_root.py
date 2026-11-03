"""Independent reconciliation checks against totals released inside source tables."""
from common import *
from root_data import MAPS,N,tab,parse_quantity
from solve_root import debt_at,period,monthwindow,pfilter,PLANS

def audit():
 checks={}
 shops={b:pd.read_pickle(CACHE/(b+'-root.pkl')) for b in MAPS}
 for b,s in shops.items():
  assert s.orders.ord.is_unique,b
  assert abs(s.lines.rev.sum()-s.orders.rev.sum())<.001,b
  assert not (s.orders.valid&(s.orders.cancel|s.orders.returned)).any(),b
  assert s.lines.date.notna().all(),b
  checks[b+'_order_count']=len(s.orders)
 for b in ['dev_003','dev_010']:
  s=shops[b]
  assert s.pay.ord.isin(s.orders.ord).all(),b
  if b=='dev_003':
   raw=pd.concat([tab(s.path/f'BanHang_TongHop_{y}.xlsx','BanHang') for y in [2025,2026]])
   raw['ord']=raw['Đơn số'].map(text);raw=raw[raw.ord.str.match(r'^ORD\d+$')].drop_duplicates('ord')
   expected=raw.loc[~raw['Trạng thái giao'].map(N).str.contains(r'\bhuy\b')&~raw['Hình thức thanh toán'].map(N).str.contains('cod'),'Công nợ'].map(num).sum()
   r=s.orders.merge(raw[['ord','Tiền hàng trước giảm']],on='ord')
   assert (r.rev-r['Tiền hàng trước giảm'].map(num)).abs().max()<.001
  else:
   raw=pd.read_csv(s.path/'don_hang.csv',dtype=str);raw['ord']=raw['Số CT'].map(text);raw=raw[raw.ord.str.match(r'^ORD\d+$')].drop_duplicates('ord')
   raw['expected_rev']=raw['Thành tiền'].map(num)+raw['Tiền giảm'].map(num)-raw['Phí vận chuyển'].map(num)
   r=s.orders.merge(raw[['ord','expected_rev']],on='ord')
   assert (r.rev-r.expected_rev).abs().max()<.001
   expected=s.orders.loc[~s.orders.cancel&~s.orders.payment.map(N).str.contains('cod'),'debt'].sum()
  actual=debt_at(s,s.end).remaining.sum()
  assert abs(actual-expected)<.001,(b,actual,expected)
  checks[b+'_final_debt_matches_source']=rnd(actual)
  checks[b+'_reconciled_order_revenues']=len(r)
 s=shops['dev_004'];v=s.orders[s.orders.valid]
 assert (v.rev-v.total).abs().max()<.001
 assert not s.lines['prod'].eq('').any()
 checks['dev_004_reconciled_valid_order_revenues']=len(v)
 for i,(op,args) in PLANS.items():
  if op in ['prod_rev','prod_qty','price_effect','import_minus_sold']:
   s=shops[QUESTIONS[i]['business_id']]
   assert not pfilter(s.lines,args[0]).empty,(i,args)
 assert parse_quantity('5 kg')==5 and parse_quantity('1 xe 4m3')==1 and parse_quantity('2 cuộn 10m')==2
 assert num('3 m3')==3 and num('1,027tr')==1027000 and num('1,090,000.0000')==1090000
 assert rnd(2.5)==3
 assert period(QUESTIONS[652],652)==monthwindow(2026,4)
 assert period(QUESTIONS[4],4)==(pd.Timestamp('2025-11-10'),pd.Timestamp('2025-11-16'))
 checks['mixed_formats_relative_dates_and_product_mappings']=True
 return checks

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
