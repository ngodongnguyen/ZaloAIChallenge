"""Read-only normalization of six shops assigned to the primary Codex analyst."""
from common import *
from functools import lru_cache
N=lru_cache(None)(norm)
D=lru_cache(None)(dt)
def parse_quantity(v):
 """Read the numeric quantity prefix, excluding digits/`k` in unit names."""
 if not text(v):return 0.
 if isinstance(v,(int,float)):return float(v)
 m=re.match(r'^\s*([+-]?\d+(?:[.,]\d+)?)',text(v).replace('−','-'))
 if not m:raise ValueError(f'Invalid quantity: {v!r}')
 return float(m[1].replace(',','.'))
def dates(s,year=None):
 lookup={v:dt(v,year) for v in s.dropna().unique()}
 return pd.to_datetime(s.map(lookup))
def convert(df,mapping):
 out=pd.DataFrame(index=df.index)
 for k,c in mapping.items():out[k]=df[c].map(lambda v:v.strip() if isinstance(v,str) else v) if c in df else ''
 return out
MAPS={
'dev_001':{'ord':'Mã đơn hàng','date':'Ngày','prod':'Tên hàng hóa','unit':'Đơn vị tính','qty':'Số lượng','rev':'Thành tiền','status':'Trạng thái','bqty':'Số lượng quy đổi','customer':'Khách hàng'},
'dev_002':{'ord':'Mã đơn hàng','date':'Ngày đặt hàng','prod':'Tên sản phẩm','unit':'Đơn vị tính','qty':'Số lượng sản phẩm','rev':'Tổng tiền','status':'Trạng thái đơn hàng','customer':'Tên khách hàng'},
'dev_003':{'ord':'Đơn số','date':'Ngày HĐ','datetime':'Giờ tạo đơn','prod':'Hàng hóa','unit':'ĐV','qty':'S.L','rev':'Thành tiền (đ)','status':'Trạng thái giao','bqty':'SL (ĐV cơ bản)','customer':'Khách hàng (tên)','cid':'Mã khách','pid':'Barcode','cost':'Giá vốn lúc bán','total':'Tổng thanh toán','paid':'Da tra','employee':'Mã NV bán','payment':'Hình thức thanh toán','channel':'Nguồn'},
'dev_004':{'ord':'Ma don','date':'Thời gian','datetime':'Giờ','prod':'Hàng bán','unit':'dvt','qty':'SL bán','rev':'Amount','status':'Trạng thái xử lý','customer':'Khách hàng (tên)','total':'Tổng phiếu'},
'dev_009':{'ord':'Mã hóa đơn','date':'Ngày tháng','status':'Trạng thái đơn','total':'TongTien','debt':'Nợ','paid':'Đã thu','customer':'Đối tác','rev':'Giá trị hàng'},
'dev_010':{'ord':'Số CT','date':'Date','status':'Xử lý','total':'Thành tiền','debt':'Con no','paid':'Khách trả','cid':'Mã thẻ','payment':'Phương thức TT','channel':'Kênh đặt'},
}
class Shop:
 def __init__(self,b):
  self.biz=b;self.path=DATA/'data'/b;p=self.path
  self.catalog=None;self.clients=None;self.pay=None;self.imports=None;self.price=None
  if b=='dev_001':
   raw=tab(p/'So_chi_tiet_ban_hang.xlsx').drop_duplicates()
   self.catalog=convert(tab(next(p.glob('Danh*.xlsx'))),{'pid':'Mã hàng hóa','prod':'Tên hàng hóa','group':'Nhóm hàng hóa','baseunit':'Đơn vị tính','basecost':'Giá vốn'})
  elif b=='dev_002':raw=pd.concat([tab(f).drop_duplicates() for f in sorted(p.glob('BanHang*.xlsx'))],ignore_index=True)
  elif b=='dev_003':
   raw=pd.concat([tab(p/f'BanHang_TongHop_{y}.xlsx','BanHang').drop_duplicates() for y in [2025,2026]],ignore_index=True)
   self.catalog=convert(tab(p/'BanHang_TongHop_2025.xlsx','BanHang_HangHoa'),{'pid':'Mã H.hóa','prod':'Tên hàng hóa','group':'Nhóm hàng','baseunit':'DVT','basecost':'Giá vốn hiện tại'})
   self.clients=convert(tab(p/'BanHang_TongHop_2025.xlsx','BanHang_KhachHang'),{'cid':'Mã KH mua','customer':'Khách hàng mua','cgroup':'Kiểu khách'})
   self.pay=convert(tab(p/'BanHang_PhieuThu.xlsx'),{'ord':'Ma don','date':'Ngày TT','amount':'Tiền thu','payment':'PTTT'})
   self.imports=convert(tab(p/'SoTay_NhapHang.xlsx'),{'iid':'Số phiếu','prod':'Product','supplier':'Nhà cung cấp','amount':'Thành tiền','note':'Lưu ý','price':'Giá nhập','qty':'SL nhập','unit':'ĐVT nhập','date':'Ngày nhập'})
   self.expenses=convert(tab(p/'SoTay_ChiPhi.xlsx'),{'date':'Ngày mua','amount':'Tiền chi'})
   self.price=convert(tab(p/'BanHang_TongHop_2025.xlsx','BanHang_LichSuGia'),{'old':'Giá cũ','pid':'Mã H.hóa','date':'Ngày áp dụng','new':'Giá mới','prod':'Item'})
  elif b=='dev_004':
   frames=[]
   for s in book(p/'SoBanHang.xlsx'):
    df=tab(p/'SoBanHang.xlsx',s).drop_duplicates()
    df['Ma don']=df['Ma don'].ffill()
    cols=['Khách hàng (tên)','Thời gian','Trạng thái xử lý','Tổng phiếu']
    df[cols]=df.groupby('Ma don')[cols].ffill()
    df['Thời gian']=dates(df['Thời gian'],int(s[-4:]));frames.append(df)
   raw=pd.concat(frames,ignore_index=True)
   self.imports=convert(tab(p/'SoNhapHang.xlsx'),{'iid':'Số phiếu','prod':'Product','supplier':'Nhà cung cấp','amount':'Thành tiền','note':'Ghi chú','price':'Giá nhập','qty':'SL nhập','unit':'ĐVT','date':'Ngày nhập'})
   self.price=convert(tab(p/'BangGia.xlsx'),{'old':'Giá cũ','date':'Ngày áp dụng','new':'Giá mới','prod':'Diễn giải'})
  elif b=='dev_009':raw=pd.read_csv(p/'danh_sach_hoa_don.csv',dtype=str).drop_duplicates()
  elif b=='dev_010':
   raw=pd.read_csv(p/'don_hang.csv',dtype=str).drop_duplicates()
   self.catalog=convert(pd.read_csv(p/'san_pham.csv',dtype=str),{'pid':'Mã vạch','prod':'ten_hang','group':'Nhóm hàng','baseunit':'Đơn vị tính'})
   self.clients=convert(pd.read_csv(p/'khach_hang.csv',dtype=str),{'cid':'Mã khách','customer':'Khách lẻ/sỉ','cgroup':'Loại khách'})
   self.pay=convert(pd.read_csv(p/'thanh_toan.csv',dtype=str),{'ord':'Mã đơn','date':'Ngay thu','amount':'So tien','payment':'Loại TT'})
  self.lines=convert(raw,MAPS[b]);l=self.lines
  l['ord']=l['ord'].map(text);l=l[l.ord.str.match(r'^ORD\d+$')].copy()
  if 'prod' in l:l=l[l['prod'].map(text).ne('')].copy()
  for c in ['rev','qty','bqty','cost','total','paid','debt']:
   if c in l:l[c]=l[c].map(parse_quantity if c=='qty' else num)
  l['date']=dates(l['date']).dt.normalize()
  if 'datetime' in l:l['datetime']=dates(l['datetime'])
  for c in ['status','unit','prod','customer','payment','channel','cid','pid','employee']:
   if c in l:l[c]=l[c].map(text)
  if b=='dev_010':
   orderfields=l.copy()
   det=convert(pd.read_csv(p/'chi_tiet_don.csv',dtype=str).drop_duplicates(),{'ord':'Mã bill','pid':'Mã H.hóa','rev':'Cộng','qty':'SoLuong','bqty':'SL quy đổi','unit':'dvt','line':'No.'})
   for c in ['ord','pid']:det[c]=det[c].map(text)
   det=det[det.ord.str.match(r'^ORD\d+$')].copy()
   for c in ['rev','qty','bqty']:det[c]=det[c].map(parse_quantity if c=='qty' else num)
   l=det.merge(l,on='ord',how='left',validate='many_to_one')
  if self.catalog is not None:
   cat=self.catalog.copy()
   cat['pid']=cat['pid'].map(text)
   if 'pid' not in l:
    cat['pkey']=cat['prod'].map(N);l['pkey']=l['prod'].map(N)
    l=l.merge(cat.drop(columns='prod'),on='pkey',how='left',validate='many_to_one')
   else:l=l.drop(columns=['prod'],errors='ignore').merge(cat,on='pid',how='left',validate='many_to_one')
   assert l.pid.notna().all(),l[l.pid.isna()].head()
  if 'pid' not in l and 'prod' in l:l['pid']=l['prod'].map(N)
  if b=='dev_002':l=self.split_toppings(l)
  if b=='dev_004':
   units=convert(tab(p/'TonKho.xlsx'),{'prod':'Tên sản phẩm','baseunit':'ĐV gốc'});units['pid']=units['prod'].map(N)
   l=l.merge(units.drop(columns='prod'),on='pid',how='left',validate='many_to_one')
   conv=tab(p/'DanhMuc.xlsx');lookup={(N(r['Hàng']),N(r['Loại ĐV'])):num(r['Hệ số']) for _,r in conv.iterrows()}
   for size in [32,50,90]:lookup[(N(f'Ống PVC Bình Minh D{size} cây 4m'),'bo')]=10
   l['factor']=[lookup.get((a,N(u)),1) for a,u in zip(l.pid,l.unit)]
   l['bqty']=l.qty*l.factor
  if b=='dev_002':
   conv=tab(p/'DanhSachQuyDoi.xlsx');lookup={(N(r['Tên sản phẩm']),N(r['Đơn vị quy đổi'])):num(r['Tỉ lệ quy đổi']) for _,r in conv.iterrows()}
   bases={N(r['Tên sản phẩm']):text(r['Đơn vị cơ bản']) for _,r in conv.iterrows()}
   l['factor']=[lookup.get((a,N(u)),1) for a,u in zip(l.pid,l.unit)]
   l['bqty']=l.qty*l.factor;l['baseunit']=[bases.get(a,u) for a,u in zip(l.pid,l.unit)]
  if self.clients is not None:
   l=l.merge(self.clients[['cid','cgroup']],on='cid',how='left',validate='many_to_one')
  if 'cid' not in l:
   l['cid']=l['customer'].map(N);l.loc[l.cid.isin(['le','kl','khach le','khach vang lai','vang lai','']), 'cid']=''
  l['cid']=l['cid'].fillna('')
  st=l['status'].map(N)
  l['cancel']=st.str.contains(r'\bhuy\b|cancel|khong giao duoc')
  l['returned']=st.str.contains(r'tra hang|hoan tra|returned|tra lai|hoan hang|hang tra ve|doi tra|khach tra')
  if 'qty' in l:l['returned']|=l.qty.lt(0)
  ret=l.groupby('ord').returned.any();cancel=l.groupby('ord').cancel.any()
  l['returned']=l.ord.map(ret);l['cancel']=l.ord.map(cancel)
  l['valid']=~l.returned & ~l.cancel
  if 'cost' in l:l['profit']=l.rev-l.qty*l.cost
  self.lines=l
  agg={'date':'first','status':'first','cid':'first','cancel':'max','returned':'max','valid':'max','rev':'sum'}
  for c in ['total','paid','debt','datetime','customer','payment','channel','employee','cgroup']:
   if c in l:agg[c]='first'
  self.orders=l.groupby('ord',sort=False).agg(agg).reset_index()
  self.start=self.orders.date.min();self.end=self.orders.date.max()
  for frame in [self.imports,self.pay,getattr(self,'expenses',None),self.price]:
   if frame is not None:
    frame['date']=dates(frame['date']).dt.normalize()
    for c in ['amount','price','qty','old','new']:
     if c in frame:frame[c]=frame[c].map(parse_quantity if c=='qty' else num)
    if 'prod' in frame:frame['pkey']=frame['prod'].map(N)
  print(b,'lines',len(l),'orders',len(self.orders),'dates',self.start,self.end,'status',sorted(set(st)),flush=True)
 def split_toppings(self,l):
  # Each included topping has its own line quantity and its own amount.
  extra=[]
  for ix in l.index[l['prod'].str.contains(' + ',regex=False)]:
   r=l.loc[ix].copy();parts=r['prod'].split(' + ');l.at[ix,'prod']=parts[0];l.at[ix,'pid']=N(parts[0])
   for s in parts[1:]:
    m=re.fullmatch(r'(?:(\d+)\s+)?(.+?)\s*\(([^()]*)\)',s)
    if not m:raise ValueError(s)
    qty=float(m[1] or 1);name=m[2];amount=num(m[3]);t=r.copy();t['prod']='Topping '+name;t['pid']=N(t['prod']);t['qty']=qty;t['rev']=amount;t['unit']='phần';extra.append(t)
    l.at[ix,'rev']-=amount
  return pd.concat([l,pd.DataFrame(extra)],ignore_index=True) if extra else l
 def select(self,a,z,orders=False,valid=True):
  df=self.orders if orders else self.lines
  mask=df.date.between(pd.Timestamp(a),pd.Timestamp(z))
  if valid:mask &=df.valid
  return df[mask]
