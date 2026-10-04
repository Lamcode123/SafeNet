from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
SEED=20261002


def wilson(k,n,z=1.959963984540054):
    if n==0: return (None,None)
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return center-half,center+half


def exact_mcnemar(b,c):
    # b = A correct, B wrong; c = A wrong, B correct. Two-sided exact binomial p.
    n=b+c
    if n==0: return 1.0
    k=min(b,c)
    prob=sum(math.comb(n,i)*(0.5**n) for i in range(k+1))
    return min(1.0,2*prob)


def holm_adjust(pairs):
    ordered=sorted(enumerate(pairs),key=lambda x:x[1])
    out=[None]*len(pairs); running=0.0; m=len(pairs)
    for rank,(idx,p) in enumerate(ordered):
        adj=min(1.0,(m-rank)*p); running=max(running,adj); out[idx]=running
    return out


def metric_pair(y,p):
    return float((y==p).mean()),float(f1_score(y,p,zero_division=0)),float(matthews_corrcoef(y,p))


def bootstrap_delta(y,pa,pb,nboot=5000):
    rng=np.random.default_rng(SEED); n=len(y); vals=[]
    for _ in range(nboot):
        ix=rng.integers(0,n,n)
        ya=y[ix]; a=pa[ix]; b=pb[ix]
        acca,f1a,mcca=metric_pair(ya,a); accb,f1b,mccb=metric_pair(ya,b)
        vals.append((acca-accb,f1a-f1b,mcca-mccb))
    arr=np.asarray(vals)
    return {
        'delta_accuracy_ci95':[float(x) for x in np.percentile(arr[:,0],[2.5,97.5])],
        'delta_f1_ci95':[float(x) for x in np.percentile(arr[:,1],[2.5,97.5])],
        'delta_mcc_ci95':[float(x) for x in np.percentile(arr[:,2],[2.5,97.5])],
    }


def main():
    pred_path=RESULTS/'ablation_predictions.csv'
    if not pred_path.exists(): raise SystemExit('Run 06_run_ablation.py first')
    df=pd.read_csv(pred_path)
    configs=sorted(df.config.unique())

    cis=[]
    for config in configs:
        g=df[(df.config==config)&df.pred_label.isin(['BENIGN','SCAM'])]
        y=g.true_label.astype(int).values; p=g.pred_label.map({'BENIGN':0,'SCAM':1}).astype(int).values
        k=int((y==p).sum()); lo,hi=wilson(k,len(y))
        cis.append({'config':config,'n':len(y),'accuracy':k/len(y) if len(y) else None,'accuracy_ci95_low':lo,'accuracy_ci95_high':hi})
    pd.DataFrame(cis).to_csv(RESULTS/'accuracy_wilson_ci.csv',index=False,encoding='utf-8-sig')

    if 'A4' not in configs:
        print('A4 absent; only Wilson CI generated.'); return

    base=df[df.config=='A4'][['sample_id','true_label','pred_label']].rename(columns={'pred_label':'pred_A4'})
    comps=[]; raw_ps=[]
    for c in configs:
        if c=='A4': continue
        other=df[df.config==c][['sample_id','pred_label']].rename(columns={'pred_label':'pred_other'})
        m=base.merge(other,on='sample_id',how='inner')
        m=m[m.pred_A4.isin(['BENIGN','SCAM']) & m.pred_other.isin(['BENIGN','SCAM'])]
        y=m.true_label.astype(int).values
        a=m.pred_A4.map({'BENIGN':0,'SCAM':1}).astype(int).values
        b=m.pred_other.map({'BENIGN':0,'SCAM':1}).astype(int).values
        ac=(a==y); bc=(b==y)
        n10=int((ac & ~bc).sum()); n01=int((~ac & bc).sum())
        pval=exact_mcnemar(n10,n01); raw_ps.append(pval)
        da,fa,ma=metric_pair(y,a); db,fb,mb=metric_pair(y,b)
        boot=bootstrap_delta(y,a,b)
        comps.append({
            'comparison':f'A4 vs {c}','n':len(y),'A4_accuracy':da,'other_accuracy':db,
            'delta_accuracy':da-db,'delta_f1':fa-fb,'delta_mcc':ma-mb,
            'mcnemar_A4_correct_other_wrong':n10,'mcnemar_A4_wrong_other_correct':n01,
            'mcnemar_p_raw':pval,**boot,
        })
    adj=holm_adjust(raw_ps)
    for r,p in zip(comps,adj): r['mcnemar_p_holm']=p
    out=pd.DataFrame(comps)
    # stringify CI arrays for CSV portability
    for col in ['delta_accuracy_ci95','delta_f1_ci95','delta_mcc_ci95']:
        out[col]=out[col].apply(json.dumps)
    out.to_csv(RESULTS/'statistical_comparison.csv',index=False,encoding='utf-8-sig')
    print(out.to_string(index=False))

if __name__=='__main__': main()
