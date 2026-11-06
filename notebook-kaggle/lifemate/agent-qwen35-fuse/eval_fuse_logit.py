import json,sys,hashlib,random,difflib,math,collections as C
import numpy as np
sys.path.insert(0,sys.argv[1])
from asr_fuse import *
ds,wc,gc,cat=sys.argv[2:6]
def cache(p):
    o={}
    for l in open(p,encoding='utf-8'):
        r=json.loads(l)
        if 'id' in r:o[r['id']]=r['text']
    return o
W,G=cache(wc),cache(gc)
rows=[json.loads(l) for l in open(ds+'/train.jsonl',encoding='utf-8')]
gold={};fin=set();ctx={};fit=[];apps={}
def walk(x,o):
    if isinstance(x,dict):
        for v in x.values(): walk(v,o)
    elif isinstance(x,list):
        for v in x: walk(v,o)
    elif isinstance(x,str) and len(x)<=80: o.append(x)
for r in rows:
    fin.add(r['audio']);aud=[r['audio']]+[t['audio'] for t in r.get('history',[]) if t.get('role')=='user']
    tx=[r.get('audio_text')]+[t.get('audio_text') for t in r.get('history',[]) if t.get('role')=='user']
    for a,t in zip(aud,tx): gold[a]=t
    if not any(a in W for a in aud): fit+=[t for t in tx if t]
    o=[];walk((r.get('context') or {}).get('user_profile'),o)
    for t in r.get('history',[]):
        if t.get('role')!='user':
            if t.get('text'):o.append(t['text'])
            walk(t.get('tool_results'),o)
    v=set(w for s in o for w in spoken_text(s))
    for a in aud: ctx[a]=v;apps[a]=r['context'].get('installed_apps',{})
C_=json.load(open(cat,encoding='utf-8'))
apptext={};cat_texts=[]
for a in C_['apps']:
    t=[a.get('name_vi',''),a.get('description','')]
    for x in a['tools']:
        t.append(x.get('description',''))
        for k,v in (x.get('parameters') or {}).get('properties',{}).items(): t.append(v.get('description',''))
    apptext[a['id']]=set(w for s in t for w in spoken_text(s));cat_texts+=[s for s in t if s]
ids=sorted(set(W)&set(G)&set(gold))
bk=lambda a:int(hashlib.md5(a.encode()).hexdigest(),16)%2
tune=[a for a in ids if not bk(a)];test=[a for a in ids if bk(a)]
GT={a:spoken_text(gold[a]) for a in ids}
def ed(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        p,d[0]=d[0],i
        for j,y in enumerate(b,1):
            p,d[j]=d[j],min(d[j]+1,d[j-1]+1,p+(x!=y))
    return d[-1]
def wer(sel,T): return sum(ed(GT[a],T[a]) for a in sel)/sum(len(GT[a]) for a in sel)
SW={a:spoken_text(W[a]) for a in ids};SG={a:spoken_text(G[a]) for a in ids}
print('W',round(wer(test,SW),4),'G',round(wer(test,SG),4))
def spans(a):
    x,y=SW[a],SG[a];out=[]
    for op,i,j,u,v in difflib.SequenceMatcher(None,x,y,autojunk=False).get_opcodes():
        out.append((op,i,j,u,v))
    return x,y,out
def build(lm,extra=None):
    Ctx=lambda a:ctx.get(a,set())
    def feats(a,left,A,B,right):
        vocab=lm.uni
        sa,sb=lm.score(left,A,right),lm.score(left,B,right)
        ca,cb=sum(w in Ctx(a) for w in A),sum(w in Ctx(a) for w in B)
        ap=set().union(*[apptext.get(k,set()) for k in apps.get(a,{})]) if apps.get(a) else set()
        ka,kb=sum(w in ap for w in A),sum(w in ap for w in B)
        oa,ob=sum(vocab[w]==0 for w in A),sum(vocab[w]==0 for w in B)
        fa=sum(math.log(1+vocab[w]) for w in A)/max(len(A),1);fb=sum(math.log(1+vocab[w]) for w in B)/max(len(B),1)
        return [sa-sb,(sa/max(len(A)+2,1))-(sb/max(len(B)+2,1)),len(A)-len(B),ca-cb,ka-kb,oa-ob,fa-fb,float(not A),float(not B),1.0]
    return feats
def collect(sel,feats):
    X=[];Y=[]
    for a in sel:
        x,y,ops=spans(a);left=['<s>','<s>'];out=[]
        for op,i,j,u,v in ops:
            if op=='equal': out+=x[i:j];left=(left+x[i:j]);continue
            A,B=x[i:j],y[u:v];right=x[j:j+2]+(['</s>'] if j+2>len(x) else [])
            g=GT[a];da,db=ed(g,out+A+x[j:]),ed(g,out+B+x[j:])
            f=feats(a,left,A,B,right)
            if da!=db: X.append(f);Y.append(1.0 if db<da else 0.0)
            pick=A if da<=db else B
            out+=pick;left=left+pick
    return np.array(X),np.array(Y)
def fit_log(X,Y,l2=1.0,it=400):
    mu,sd=X.mean(0),X.std(0)+1e-6;sd[-1]=1;mu[-1]=0;Z=(X-mu)/sd;w=np.zeros(Z.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-Z@w));g=Z.T@(p-Y)/len(Y)+l2*w/len(Y);H=(Z*(p*(1-p))[:,None]).T@Z/len(Y)+np.eye(len(w))*(l2/len(Y)+1e-6);w-=np.linalg.solve(H,g)
    return w,mu,sd
def apply(sel,feats,model):
    w,mu,sd=model;T={}
    for a in sel:
        x,y,ops=spans(a);left=['<s>','<s>'];out=[]
        for op,i,j,u,v in ops:
            if op=='equal': out+=x[i:j];left=left+x[i:j];continue
            A,B=x[i:j],y[u:v];right=x[j:j+2]+(['</s>'] if j+2>len(x) else [])
            z=(np.array(feats(a,left,A,B,right))-mu)/sd
            pick=B if z@w>0 else A
            out+=pick;left=left+pick
        T[a]=out
    return T
def rule(sel,lm,b=-2,c=2,l=1):
    T={}
    for a in sel:
        x,y,ops=spans(a);left=['<s>','<s>'];out=[]
        for op,i,j,u,v in ops:
            if op=='equal': out+=x[i:j];left=left+x[i:j];continue
            A,B=x[i:j],y[u:v];right=x[j:j+2]+(['</s>'] if j+2>len(x) else [])
            tot=lambda z:lm.score(left,z,right)+c*sum(w in ctx.get(a,set()) for w in z)-l*len(z)
            pick=A if tot(A)+b>=tot(B) else B
            out+=pick;left=left+pick
        T[a]=out
    return T
def report(name,T,base=None):
    print(f'{name:28} test {wer(test,T):.4f} final {wer([a for a in test if a in fin],T):.4f}')
lm0=TrigramLM(fit);lm1=TrigramLM(fit+cat_texts)
report('rule LM=train',rule(test,lm0));report('rule LM=train+catalog',rule(test,lm1))
for name,lm in (('logit LM=train',lm0),('logit LM=train+catalog',lm1)):
    f=build(lm);X,Y=collect(tune,f);m=fit_log(X,Y)
    print(' spans',len(Y),'pick-gip rate',round(Y.mean(),3),'train acc',round((((( (X-m[1])/m[2])@m[0])>0)==(Y>0.5)).mean(),3),'weights',np.round(m[0],2).tolist())
    T=apply(test,f,m);report(name,T)
    dd={a:(ed(GT[a],SW[a]),ed(GT[a],T[a]),len(GT[a])) for a in test}
    R=rule(test,lm0);d2={a:ed(GT[a],R[a]) for a in test}
    rng=random.Random(0);ks=list(dd);bt=[]
    for _ in range(1000):
        s=rng.choices(ks,k=len(ks));n=sum(dd[k][2] for k in s);bt.append(100*(sum(d2[k] for k in s)-sum(dd[k][1] for k in s))/n)
    bt.sort();print(' gain over current rule pts',round(100*(sum(d2.values())-sum(v[1] for v in dd.values()))/sum(v[2] for v in dd.values()),2),'CI95',round(bt[25],2),round(bt[975],2))
orc={}
for a in test:
    x,y,ops=spans(a);out=[];g=GT[a]
    for op,i,j,u,v in ops:
        if op=='equal': out+=x[i:j];continue
        A,B=x[i:j],y[u:v];out+=A if ed(g,out+A+x[j:])<=ed(g,out+B+x[j:]) else B
    orc[a]=out
report('slot oracle (trần)',orc)
