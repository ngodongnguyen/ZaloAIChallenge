"""Prototype: choose the paying source from balances the agent has already read. Keeps the proposal unless certain."""
import re
A=re.compile(r'\bví\b.{0,40}(đủ|thiếu)|(đủ|thiếu).{0,30}\bví\b',re.I)
B=re.compile(r'(đủ|thiếu).{0,50}(tiết kiệm)|(tiết kiệm).{0,50}(đủ|thiếu)',re.I)
WORDS={'electricity':['tiền điện'],'water':['tiền nước'],'internet':['tiền mạng','tiền net','tiền internet'],'tv':['tiền truyền hình','tiền tv','tiền ti vi','tiền tivi'],'phone':['tiền điện thoại','cước điện thoại']}
def seen(steps,name):
    out=None
    for s in steps:
        if s['name']==name: out=s['result']
    return out
def rule(request,steps,name,args):
    t=request.lower()
    if name=='bill_payment.pay' and A.search(t) and args.get('source') in('ewallet','bank'):
        bal=seen(steps,'ewallet.get_balance');bills=seen(steps,'bill_payment.list_bills')
        if not bal or not bills or 'balance' not in bal:return args
        amt=next((b['amount'] for b in bills.get('bills',[]) if b['id']==args.get('bill_id')),None)
        if amt is None:return args
        return dict(args,source='ewallet' if bal['balance']>=amt else 'bank')
    if name=='banking.transfer' and B.search(t) and 'thanh toán' in t and not re.search('chia|nửa|một phần',t):
        bal=seen(steps,'banking.get_balance');bills=seen(steps,'bill_payment.list_bills')
        if not bal or not bills:return args
        acc={a['name'].lower():a for a in bal.get('accounts',[])}
        pay=next((a for n,a in acc.items() if 'thanh toán' in n),None);sav=next((a for n,a in acc.items() if 'tiết kiệm' in n),None)
        if not pay or not sav:return args
        total=sum(b['amount'] for b in bills.get('bills',[]) if any(w in t for w in WORDS.get(b.get('type'),[])))
        done=sum(1 for s in steps if s['name']=='banking.transfer')
        if done or not total:  # later transfers follow the account the first one used
            prev=[s for s in steps if s['name']=='banking.transfer']
            return dict(args,from_account_id=prev[0]['arguments'].get('from_account_id')) if prev else args
        return dict(args,from_account_id=pay['id'] if pay['balance']>=total else sav['id'])
    return args
