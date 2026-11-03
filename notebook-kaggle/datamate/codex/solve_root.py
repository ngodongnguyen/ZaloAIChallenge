"""Direct question-by-question Codex query plans and computed answers for six shops."""
from root_data import *
import calendar,collections,traceback

def plain(v):
 s=unicodedata.normalize('NFD',text(v).lower()).replace('đ','d')
 return ''.join(c for c in s if unicodedata.category(c)!='Mn')
def monthwindow(y,m):
 return pd.Timestamp(y,m,1),pd.Timestamp(y,m,calendar.monthrange(y,m)[1])
def period(q,i):
 s=plain(q['question']);n=norm(s);d=pd.Timestamp(q['as_of_date'])
 if i==652:return monthwindow(2026,4)
 if i in [112,155,889]:return pd.Timestamp('2026-02-10'),pd.Timestamp('2026-02-26')
 if 'tuan truoc' in n or 'tuan roi' in n:
  a=d-pd.Timedelta(days=d.weekday()+7);return a,a+pd.Timedelta(days=6)
 if 'tuan nay' in n or 'tuan ni' in n:return d-pd.Timedelta(days=d.weekday()),d
 if any(x in n for x in ['dau nam','ngay 1 1']):return pd.Timestamp(d.year,1,1),d
 m=re.search(r'(7|30)\s*ngay',s)
 if m:return d-pd.Timedelta(days=int(m[1])-1),d
 m=re.search(r'quy\s*([1-4])(?:\s*(?:/|nam)?\s*(202[56]|2[56]))?',s)
 if m:
  y=int(m[2]) if m[2] else d.year;y=2000+y if y<100 else y;mo=(int(m[1])-1)*3+1
  if not m[2] and mo>d.month:y-=1
  return monthwindow(y,mo)[0],monthwindow(y,mo+2)[1]
 matches=list(re.finditer(r'(?<!\d)(?:thang\s*|th\s*|t\s*)?(0?[1-9]|1[0-2])\s*[/\-]\s*(202[56]|2[56])\b',s))
 if matches:
  m=matches[0];y=int(m[2]);y=2000+y if y<100 else y;return monthwindow(y,int(m[1]))
 m=re.search(r'(?:thang|th|t)\s*(0?[1-9]|1[0-2])(?:\s+nam\s+|\s+)(202[56])\b',s)
 if m:return monthwindow(int(m[2]),int(m[1]))
 if i==926:return monthwindow(2026,3)
 m=re.search(r'(?:thang|th)\s*(0?[1-9]|1[0-2])\b',s)
 if m:
  mo=int(m[1]);return monthwindow(d.year-(mo>d.month),mo)
 if any(t in n for t in ['thang nay','th nay','thang ni','th ni']):return pd.Timestamp(d.year,d.month,1),d
 if any(t in n for t in ['hom qua','bua qua','hqua']):return d-pd.Timedelta(days=1),d-pd.Timedelta(days=1)
 if any(t in n for t in ['hom nay','hnay','hom ni','bua nay']):return d,d
 raise ValueError(f'No period {i}: {s}')

PLANS={}
def add(op,indices,*args):
 for i in indices:
  assert i not in PLANS,i
  PLANS[i]=(op,args)
# dev_001
add('rev',[223,284,298,407,536,620,680,769,855])
add('orders',[38,65,77,135,574,658])
add('avg',[41,456,467,668,914,975])
add('cancel',[587,701,723]);add('cancelrate',[184])
add('growth_month',[201,663,667]);add('growth_year',[381,548,764]);add('growth_mtd',[150,355,880])
add('bestday',[342]);add('cust_orders',[26,470,622]);add('cust_rev',[726])
add('cust_min',[62],4);add('cust_min',[863],5);add('cust_min',[906],3)
add('cust_min_share',[92,752],3);add('cust_min_share',[449],5)
add('order_threshold',[103],180000);add('order_threshold',[338],190000);add('order_threshold',[718],400000)
add('day_threshold',[312],9500000);add('day_threshold',[332],18000000);add('day_threshold',[953],10000000)
add('product_threshold',[144],84);add('product_threshold',[145],74);add('product_threshold',[974],79)
add('top_products',[141,954],3)
add('prod_rev',[9],'Bia Heineken 330ml');add('prod_rev',[829],'Bia Sài Gòn Special 330ml');add('prod_rev',[850],'Sữa đặc Ông Thọ 380g')
add('group_rev',[516],'thực phẩm');add('group_rev',[746],'bánh kẹo');add('group_share',[700],'hoá phẩm')
add('group_orders',[545],'bia');add('group_orders',[646],'bánh kẹo');add('group_orders',[719],'đồ ăn vặt')
add('group_top_qty',[649],'gạo');add('group_top_rev',[766],'gia vị');add('group_top_rev',[802],'chăm sóc cá nhân')
add('unit_qty',[596,780],'gói');add('unit_qty',[755],'chai')
add('unanswerable',[117,369,459,775],'Dữ liệu không ghi kênh bán/giới tính/địa chỉ/lịch sử giá hay tiêu chí chất lượng cần thiết cho câu hỏi.')
add('price_effect',[218],'Đậu phộng rang muối 200g',21000,30)
add('price_effect',[787],'Dầu gội Clear 630g',152000,30)
add('price_effect',[853],'Sữa đặc Ông Thọ 380g',26000,30)
# dev_002
add('rev',[2,118,402,478,758,859,948,993]);add('orders',[17,354,420,614]);add('avg',[19,21,192])
add('cancel',[53,895,900]);add('cancelrate',[44,357,763])
add('growth_year',[153,161,652]);add('growth_month',[263,311,847]);add('bestday',[277,934])
add('top_products',[335,651],3);add('cust_top_rev',[920],3);add('loyal',[473])
add('prod_qty',[64],'Nước ép cam (M)');add('prod_qty',[691],'Cà phê sữa (M)');add('prod_qty',[870],'Latte (M)')
add('size_qty',[776],'M');add('size_qty',[814],'L')
add('unanswerable',[0,262,675],'File bán hàng và bảng quy đổi không ghi nhóm hàng; không suy nhóm từ tên sản phẩm theo rulebook §7.')
# dev_003
add('rev',[112,166,175,282,285,458,644,645,685,866]);add('orders',[102,107,146,174,314,512]);add('avg',[6,39,288,322,487,575])
add('profit',[69,274,419]);add('netprofit',[31,393,971]);add('bestprofit',[33,204,773])
add('cancel',[535,926,930]);add('growth_month',[225,607,875]);add('growth_year',[165,565]);add('growth_mtd',[387,494,580]);add('bestday',[960])
add('cust_orders',[55]);add('cust_min',[294,488],9) # corrected below for index488
PLANS[488]=('cust_min',(11,))
add('cust_min',[418],12);add('cust_min_share',[424,703],12);add('cust_min_share',[539],11)
add('order_threshold',[569],52000);add('order_threshold',[629],130000);add('order_threshold',[929],55000)
add('day_threshold',[196],7600000);add('day_threshold',[271],9100000);add('day_threshold',[491],8900000)
add('product_threshold',[84],122);add('product_threshold',[594],152);add('product_threshold',[722],128)
add('top_products',[708,888],5);add('top_products_one',[889])
add('prod_qty',[252],'Bánh tiramisu');add('prod_qty',[413],'Bánh croissant');add('prod_qty',[706],'Topping Thạch cà phê')
add('prod_rev',[358],'Cà phê muối (M)');add('prod_rev',[812],'Topping Trân châu đen')
add('group_qty',[336],'cà phê');add('group_qty',[586],'trà sữa');add('group_rev',[835],'trà')
add('group_share',[152],'nước ép');add('group_share',[206],'đá xay');add('group_orders',[537],'trà')
add('group_top_qty',[412],'sinh tố');add('group_top_rev',[437],'bánh')
add('size_qty',[91],'M');add('size_qty',[212],'L');add('unit_qty',[762,808],'ly')
add('cgroup_avg',[83],'thân thiết');add('cgroup_rev',[729],'thân thiết');add('cgroup_share',[965],'thân thiết')
add('employee_rev',[148],'NV005');add('employee_rev',[531],'NV001');add('employee_rev',[949],'NV004')
add('shift_orders',[99],'morning');add('shift_rev',[194],'afternoon');add('shift_rev',[584],'morning')
add('shift_growth_orders',[256]);add('shift_growth_rev',[378])
add('supplier_main',[36]);add('supplier_money',[169]);add('import_minus_sold',[704],'Bánh croissant');add('import_minus_sold',[810],'Bánh tiramisu')
add('debt_count',[231,588,720]);add('debt_top',[241,513,981]);add('debt_older',[261,465,590],7);add('debt_ages',[362,502,662])
add('debt_flows_customer',[237],'Hà Duy Trung');add('debt_flows_customer',[475],'Lưu Sơn Hằng');add('debt_repaid',[485,711,747])
add('channel_rev',[421],'store');add('payment_rev',[805],'chuyển khoản');add('payment_pair',[992],'ví điện tử','tiền mặt')
add('price_effect',[264],'Trà vải (L)',47000,30);add('price_effect',[785],'Topping Thạch cà phê',7400,30);add('price_effect',[864],'Bạc xỉu (M)',35000,14)
add('tet_growth',[155])
add('unanswerable',[457,656,709],'Dữ liệu không ghi tính trung thực của khách hoặc lý do cụ thể huỷ/trả hàng.')
# dev_004
add('rev',[4,210,219,462,507,682,753,846,958]);add('orders',[149,392,468,576,632,935]);add('avg',[279,379,572,748,858,997])
add('cancel',[400,401]);add('cancelrate',[635,943]);add('growth_month',[229,730,851]);add('growth_year',[443,690]);add('growth_mtd',[132,533,674]);add('bestday',[328])
add('cust_orders',[434,840,918]);add('cust_min',[404],2);add('cust_min',[463,955],4);add('cust_min_share',[14],4);add('cust_min_share',[95,616],3)
add('prod_qty',[1],'Ống PVC Bình Minh D50 cây 4m');add('prod_qty',[11],'Ống PVC Bình Minh D32 cây 4m');add('prod_qty',[768],'Ống PVC Bình Minh D90 cây 4m')
add('order_threshold',[666],1800000);add('order_threshold',[721],550000);add('order_threshold',[867],3700000)
add('day_threshold',[524],40000000);add('day_threshold',[760],43000000);add('day_threshold',[902],39000000)
add('top_products',[168,409],5);add('top_products_one',[461]);add('peak_hour',[123])
add('shift_growth_rev',[249]);add('shift_growth_orders',[296,911]);add('shift_rev',[824],'morning');add('shift_orders',[861],'afternoon')
add('dead_products',[181,669,809]);add('import_total',[380]);add('supplier_money',[807]);add('delivery_stages',[677,783,788]);add('delivery_two',[375,505,589]);add('delivery_qty',[363],'Dây điện Cadivi VCmd 2x2.5 cuộn 100m',1);add('delivery_qty',[483],'Sơn Dulux Weathershield Powerflexx 5L',1);add('delivery_qty',[811],'Gạch ốp tường Viglacera 30x60 thùng 8 viên',1)
add('due_future',[74,108,624]);add('due_month',[471],2026,4);add('due_month',[637],2025,11);add('due_month',[681],2026,2)
add('price_effect',[105],'Nước kiềm Ion Life 19L',79500,30);add('price_effect',[894],'Gas Petrolimex 12kg',546000,30);add('price_effect',[963],'Cát xây tô',315000,14)
add('unanswerable',[122,191,915,928,988],'Sổ không ghi lý do huỷ, địa chỉ khách hay nhóm hàng cần thiết; không tự suy ra từ tên theo rulebook §7.')
# dev_009
add('rev',[47,50,171,376,544,551,636,671,754,771,821]);add('orders',[12,81,200,333,361,422,514]);add('bestday',[276,872,909])
# dev_010
add('rev',[37,287,517,521,670,731,879,901,925]);add('orders',[266,581,839,849,921,956]);add('avg',[202,344,642]);add('cancel',[100,350,697]);add('cancelrate',[162,374,673])
add('growth_month',[59,525,845]);add('growth_year',[301,334,799]);add('bestday',[179,977]);add('bestday_orders',[238])
add('prod_rev',[163],'Bao tay bao chân sơ sinh set 3 đôi');add('prod_rev',[254],'Nước giặt xả D-nee 3000ml');add('prod_rev',[820],'Trái cây nghiền HiPPiS Organic 100g')
add('group_rev',[542],'bình sữa');add('group_share',[227],'bỉm tã');add('group_share',[710],'ăn dặm')
add('cgroup_rev',[46],'sỉ');add('cgroup_rev',[528],'thân thiết');add('cgroup_share',[538],'thân thiết');add('cgroup_share',[985],'sỉ')
add('cust_rev',[577]);add('top_products',[831],5);add('top_products_one',[982]);add('unit_qty',[119,818],'gói')
add('channel_pair',[128],'store','phone');add('payment_pair',[395],'cash','ewallet');add('payment_share',[368],'transfer')
add('debt_repaid',[97,450,664]);add('debt_count',[439]);add('debt_top',[621,736]);add('debt_total',[822])
add('unanswerable',[482],'Dữ liệu không ghi tiêu chí khách thân thiện nhất.')

MULTI={23:[(2025,3),(2025,4),(2025,5)],618:[(2025,7),(2025,8),(2025,9)],688:[(2026,2),(2026,3),(2026,4)],742:[(2025,6),(2025,7),(2025,8)],796:[(2025,2),(2025,3),(2025,4)],833:[(2026,1),(2026,2),(2026,3)],886:[(2026,4),(2026,5),(2026,6)],903:[(2026,3),(2026,4),(2026,5)],931:[(2025,12),(2026,1),(2026,2)],75:[(2025,5),(2025,6),(2025,7)],403:[(2026,4),(2026,5),(2026,6)],411:[(2025,7),(2025,8),(2025,9)],476:[(2025,8),(2025,9),(2025,10)],606:[(2025,11),(2025,12),(2026,1)],792:[(2025,3),(2025,4),(2025,5)],800:[(2026,1),(2026,2),(2026,3)],899:[(2025,12),(2026,1),(2026,2)],180:[(2025,11),(2025,12),(2026,1)],583:[(2026,2),(2026,3),(2026,4)],705:[(2025,5),(2025,6),(2025,7)],727:[(2026,1),(2026,2),(2026,3)],745:[(2025,10),(2025,11),(2025,12)],838:[(2025,6),(2025,7),(2025,8)],952:[(2025,12),(2026,1),(2026,2)]}
add('multi_rev',[618,796,886,403,411,180,705,745]);add('multi_increase',[742,833,931,75,476,606,583]);add('multi_best',[23,688,903,792,800,899,727,838,952])

# All operations are based on released cells and rulebook defaults.
def rank(df,key,value,how='sum'):
 d=df.groupby(key)[value].agg(how).reset_index();return d.sort_values([value,key],ascending=[False,True],kind='stable')
def pct(a,b):
 if b==0:raise ValueError('Zero comparison denominator')
 return rnd(100*(a-b)/b,2)
def pfilter(df,name):return df[df['prod'].map(N).eq(N(name))]
def named(s,key):
 col='prod' if key=='pid' else 'customer'
 if col not in s.lines:return dict(zip(s.lines.get(key,[]),s.lines.get(key,[])))
 best={}
 for _,r in s.lines[[key,col]].dropna().drop_duplicates().iterrows():
  k=r[key];c_str=str(r[col])
  if k not in best or sum(ch>'\x7f' for ch in c_str)>sum(ch>'\x7f' for ch in best[k]):best[k]=c_str
 return best
def debt_at(s,d):
 o=s.orders[s.orders.date.le(d)&~s.orders.cancel].copy()
 if 'payment' in o:o=o[~o.payment.map(N).str.contains(r'cod|san tmdt|ecommerce|marketplace')]
 pay=s.pay[s.pay.date.le(d)].groupby('ord').amount.sum()
 o['remaining']=(o.total-o.ord.map(pay).fillna(0)).clip(lower=0)
 o['age']=(d-o.date).dt.days
 return o[o.remaining.gt(0)]
def repaid(s,a,z,cname=None):
 o=s.orders[~s.orders.cancel & s.orders.payment.map(N).isin(['debt','ghi no','no','mua chiu'])]
 if cname is not None:
  cid=s.lines.loc[s.lines['customer'].map(N).eq(N(cname)),'cid'].unique()
  o=o[o.cid.isin(cid)]
 p=s.pay.merge(o[['ord','date']].rename(columns={'date':'purchase_date'}),on='ord')
 p=p[p.date.gt(p.purchase_date) & p.date.between(a,z)]
 return rnd(p.amount.sum())
def solve(s,q,i):
 op,args=PLANS[i]
 if op=='unanswerable':return output(op,reason=args[0]),{}
 d=pd.Timestamp(q['as_of_date'])
 if op.startswith('multi_'):
  months=MULTI[i];values=[rnd(s.select(*monthwindow(y,m)).rev.sum()) for y,m in months]
  if any(monthwindow(y,m)[1]<s.start or monthwindow(y,m)[0]>s.end for y,m in months):return output('unanswerable',reason='Kỳ hỏi hoàn toàn ngoài khoảng dữ liệu.'),{'months':months}
  if op=='multi_rev':r=output('list',values)
  elif op=='multi_increase':r=output('boolean',all(y>x for x,y in zip(values,values[1:])))
  else:r=output('date','%04d-%02d'%months[values.index(max(values))])
  return r,{'months':months,'revenues':values}
 if op.startswith('debt_') and op not in ['debt_repaid','debt_flows_customer']:
  o=debt_at(s,d);ev={'snapshot':str(d.date()),'orders_with_debt':len(o)}
  if op=='debt_count':return output('number',int(o.loc[o.cid.ne(''),'cid'].nunique())),ev
  if op=='debt_total':return output('number',rnd(o.remaining.sum())),ev
  if op=='debt_older':return output('number',rnd(o.loc[o.age.gt(args[0]),'remaining'].sum())),ev
  if op=='debt_ages':return output('list',[rnd(o.loc[o.age.between(0,7),'remaining'].sum()),rnd(o.loc[o.age.between(8,15),'remaining'].sum()),rnd(o.loc[o.age.gt(15),'remaining'].sum())]),ev
  if op=='debt_top':
   k=3 if i in [513,736] else 5;r=rank(o[o.cid.ne('')],'cid','remaining').head(k);names=named(s,'cid')
   return output('table',{'columns':['khach_hang','so_tien_no'],'rows':[[names[c],rnd(v)] for c,v in zip(r.cid,r.remaining)]}),ev
 if op.startswith('due_'):
  t=tab(s.path/'SoHanNo.xlsx');t['ord']=t['Mã đơn hàng'].map(text);t['due']=dates(t['Ngày phải trả'])
  t=t.merge(s.orders[['ord','date','cancel']],on='ord');t=t[t.date.le(d)&~t.cancel]
  count=t.due.gt(d).sum() if op=='due_future' else t.due.between(*monthwindow(*args)).sum()
  return output('number',int(count)),{'as_of':str(d.date()),'eligible_orders':len(t)}
 if op=='price_effect':
  name,new,days=args
  explicit={218:'2025-09-01',787:'2025-05-28',853:'2025-11-13'}
  if i in explicit:a=pd.Timestamp(explicit[i])
  else:
   t=s.price[s.price.pkey.eq(N(name)) & s.price.new.eq(new)&s.price.old.gt(0)&s.price.new.gt(s.price.old)].sort_values('date')
   if t.empty:raise ValueError(('price',args))
   a=t.iloc[0].date
  b=a+pd.Timedelta(days=days-1);c=a-pd.Timedelta(days=days)
  x=pfilter(s.select(a,b),name).bqty.sum();y=pfilter(s.select(c,a-pd.Timedelta(days=1)),name).bqty.sum()
  return output('percent',pct(x,y)),{'change_date':str(a.date()),'after':float(x),'before':float(y),'window_days':days}
 a,z=period(q,i)
 if op=='dead_products' and i==809:a,z=pd.Timestamp('2026-02-15'),pd.Timestamp('2026-02-21')
 ev={'from':str(a.date()),'to':str(z.date())}
 if z<s.start or a>s.end:return output('unanswerable',reason='Kỳ hỏi nằm hoàn toàn ngoài khoảng ngày có dữ liệu của cửa hàng.'),ev
 l=s.select(a,z);o=s.select(a,z,orders=True);all_o=s.select(a,z,orders=True,valid=False);ev.update({'valid_orders':len(o),'valid_lines':len(l)})
 if op=='rev':return output('number',rnd(l.rev.sum())),ev
 if op=='orders':return output('number',len(o)),ev
 if op=='avg':return output('number',rnd(l.rev.sum()/len(o))),ev
 if op=='cancel':return output('number',int(all_o.cancel.sum())),ev
 if op=='cancelrate':return output('percent',rnd(100*all_o.cancel.sum()/len(all_o),2)),ev
 if op in ['growth_month','growth_year','growth_mtd']:
  if op=='growth_mtd':a,z=pd.Timestamp(d.year,d.month,1),d;prev=a-pd.offsets.MonthBegin(1);end=prev+pd.Timedelta(days=min(d.day,calendar.monthrange(prev.year,prev.month)[1])-1)
  elif op=='growth_year':prev,end=monthwindow(a.year-1,a.month)
  else:prev=a-pd.offsets.MonthBegin(1);end=a-pd.Timedelta(days=1)
  x=s.select(a,z).rev.sum();y=s.select(prev,end).rev.sum()
  if end<s.start or prev>s.end:return output('unanswerable',reason='Kỳ so sánh nằm hoàn toàn ngoài dữ liệu phát hành.'),ev
  ev.update({'from':str(a.date()),'to':str(z.date()),'current':float(x),'previous':float(y),'previous_from':str(prev.date()),'previous_to':str(end.date())})
  return output('percent',pct(x,y)),ev
 if op in ['bestday','bestday_orders']:
  r=rank(o,'date','rev' if op=='bestday' else 'ord','sum' if op=='bestday' else 'count');return output('date',str(r.iloc[0].date.date())),ev
 if op in ['cust_orders','cust_rev','cust_top_rev','loyal']:
  c=o[o.cid.ne('')];val='rev' if op in ['cust_rev','cust_top_rev'] else 'date' if op=='loyal' else 'ord';method='sum' if val=='rev' else 'nunique'
  r=rank(c,'cid',val,method);names=named(s,'cid');k=args[0] if op=='cust_top_rev' else 1
  result=[names[x] for x in r.head(k).cid]
  return output('list',result) if op=='cust_top_rev' else output('string',result[0]),ev
 if op in ['cust_min','cust_min_share']:
  c=o[o.cid.ne('')].groupby('cid').size();n=int(c.ge(args[0]).sum());return output('number',n) if op=='cust_min' else output('percent',rnd(100*n/len(c),2)),ev
 if op=='order_threshold':return output('number',int(o.rev.ge(args[0]).sum())),ev
 if op=='day_threshold':return output('number',int(o.groupby('date').rev.sum().ge(args[0]).sum())),ev
 if op=='product_threshold':return output('number',int(l.groupby('pid').ord.nunique().ge(args[0]).sum())),ev
 if op in ['top_products','top_products_one','bestprofit']:
  if 'prod' in l:l=l[~l['prod'].map(N).eq('phi phuc vu')]
  val='profit' if op=='bestprofit' else 'bqty';r=rank(l,'pid',val);names=named(s,'pid');k=args[0] if op=='top_products' else 1
  result=[names[x] for x in r.head(k).pid]
  return output('list',result) if op=='top_products' else output('string',result[0]),ev
 if op in ['prod_rev','prod_qty']:
  t=pfilter(l,args[0]);ev['product']=args[0];return output('number',rnd(t.rev.sum()) if op=='prod_rev' else rnd(t.bqty.sum(),2)),ev
 if op=='unit_qty':return output('number',rnd(l.loc[l.baseunit.map(N).eq(N(args[0])),'bqty'].sum(),2)),ev
 if op=='size_qty':return output('number',rnd(l.loc[l['prod'].str.contains(r'\('+args[0]+r'\)$',case=False),'bqty'].sum(),2)),ev
 if op.startswith('group_'):
  t=l[l.group.map(N).eq(N(args[0]))];ev['group']=args[0]
  if op=='group_rev':return output('number',rnd(t.rev.sum())),ev
  if op=='group_share':return output('percent',rnd(100*t.rev.sum()/l.rev.sum(),2)),ev
  if op=='group_orders':return output('number',int(t.ord.nunique())),ev
  if op=='group_qty':return output('number',rnd(t.bqty.sum(),2)),ev
  r=rank(t,'pid','rev' if op=='group_top_rev' else 'bqty');return output('string',named(s,'pid')[r.iloc[0].pid]),ev
 if op.startswith('cgroup_'):
  t=o[o.cgroup.map(N).eq(N(args[0]))&o.cid.ne('')]
  if op=='cgroup_rev':return output('number',rnd(t.rev.sum())),ev
  if op=='cgroup_share':return output('percent',rnd(100*t.rev.sum()/o.rev.sum(),2)),ev
  return output('number',rnd(t.rev.sum()/len(t))),ev
 if op=='employee_rev':return output('number',rnd(o.loc[o.employee.eq(args[0]),'rev'].sum())),ev
 if op.startswith('shift_'):
  hour=o['datetime'].dt.hour;morning=o[hour.between(6,13)];afternoon=o[hour.between(14,21)]
  if 'growth' in op:
   x,y=(len(afternoon),len(morning)) if op.endswith('orders') else (afternoon.rev.sum(),morning.rev.sum());return output('percent',pct(x,y)),ev
  t=morning if args[0]=='morning' else afternoon
  return output('number',len(t) if op=='shift_orders' else rnd(t.rev.sum())),ev
 if op=='peak_hour':
  r=o.assign(hour=o['datetime'].dt.hour).groupby('hour').size();return output('number',int(r.sort_index().idxmax())),ev
 if op in ['profit','netprofit']:
  val=l.profit.sum()
  if op=='netprofit':val-=s.expenses.loc[s.expenses.date.between(a,z),'amount'].sum()
  return output('number',rnd(val)),ev
 if op in ['import_total','supplier_main','supplier_money','import_minus_sold']:
  t=s.imports[s.imports.date.between(a,z) & ~s.imports.note.map(N).str.contains('huy|cancel')]
  if op=='import_total':return output('number',rnd(t.amount.sum())),ev
  if op=='import_minus_sold':
   t=t[t.pkey.eq(N(args[0]))];sold=pfilter(l,args[0]).bqty.sum();return output('number',rnd((t.qty*t.unit.map(lambda u:6 if N(u)=='hop' else 1)).sum()-sold,2)),ev
  r=rank(t,'supplier','iid' if op=='supplier_main' else 'amount','nunique' if op=='supplier_main' else 'sum');return output('string',r.iloc[0].supplier),ev
 if op=='debt_repaid':return output('number',repaid(s,a,z)),ev
 if op=='debt_flows_customer':
  name=args[0];cids=s.lines.loc[s.lines['customer'].map(N).eq(N(name)),'cid'].unique();t=all_o[~all_o.cancel & all_o.cid.isin(cids)&all_o.payment.map(N).isin(['ghi no','debt','no','mua chiu'])]
  same=s.pay.merge(t[['ord','date']].rename(columns={'date':'purchase_date'}),on='ord');same=same[same.date.eq(same.purchase_date)]
  immediate=same.groupby('ord').amount.sum();added=(t.total-t.ord.map(immediate).fillna(0)).clip(lower=0).sum()
  return output('list',[rnd(added),repaid(s,a,z,name)]),ev
 if op in ['payment_rev','payment_pair','payment_share','channel_rev','channel_pair']:
  col='payment' if op.startswith('payment') else 'channel';vals=[rnd(o.loc[o[col].map(N).eq(N(v)),'rev'].sum()) for v in args]
  if op.endswith('pair'):return output('list',vals),ev
  if op.endswith('share'):return output('percent',rnd(100*vals[0]/o.rev.sum(),2)),ev
  return output('number',vals[0]),ev
 if op=='tet_growth':
  prev=a-pd.Timedelta(days=17);before=s.select(prev,a-pd.Timedelta(days=1)).rev.sum();return output('percent',pct(l.rev.sum(),before)),ev
 if op=='dead_products':
  before=s.lines[s.lines.date.lt(a)&s.lines.valid];prior=set(before.pid);current=set(l.pid);return output('number',len(prior-current)),ev
 if op.startswith('delivery_'):
  t=convert(tab(s.path/'SoGiaoHang.xlsx'),{'ord':'Số đơn','date':'Thời gian giao','stage':'Lượt giao','prod':'Hàng hóa','qty':'Số lượng giao'});t['date']=dates(t['date']);t['qty']=t['qty'].map(num)
  if op=='delivery_stages':return output('number',int(t[t.date.between(a,z)].drop_duplicates(['ord','stage']).shape[0])),ev
  # Month in these questions describes original order date. Physical deliveries include returned orders.
  t=t[t.date.le(d)].merge(s.orders[['ord','date','cancel']].rename(columns={'date':'orderdate'}),on='ord');t=t[t.orderdate.between(a,z)&~t.cancel]
  if op=='delivery_two':return output('number',int(t.groupby('ord').stage.nunique().ge(2).sum())),ev
  return output('number',rnd(pfilter(t,args[0]).qty.sum()/args[1],2)),ev
 raise ValueError(('unknown op',op))

def main():
 answers={};evidence={};errors=[]
 shops={b:Shop(b) for b in MAPS}
 for b,s in shops.items():pd.to_pickle(s,CACHE/(b+'-root.pkl'))
 for i,q in enumerate(QUESTIONS):
  if q['business_id'] not in shops:continue
  try:
   if i not in PLANS:raise ValueError('NO PLAN')
   r,ev=solve(shops[q['business_id']],q,i);r={'id':q['id'],**r};answers[q['id']]=r;evidence[q['id']]={'index':i,'business_id':q['business_id'],'question':q['question'],'operation':PLANS[i],**ev}
  except Exception as e:errors.append((i,str(e),q['question']));traceback.print_exc()
 (OUT/'answers_root.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2,allow_nan=False))
 (OUT/'evidence_root.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,default=str))
 if errors:raise RuntimeError(f'{len(errors)} unresolved root questions: {errors}')
 print('ANSWERED',len(answers),'TYPES',collections.Counter(x['type'] for x in answers.values()),'ERRORS',len(errors))
 for e in errors:print(e)
if __name__=='__main__':main()
