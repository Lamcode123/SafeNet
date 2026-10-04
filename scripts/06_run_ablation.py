from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; RESULTS=ROOT/'results'
ENDPOINT_DEFAULT='http://127.0.0.1:8000/api/research/verify'
HEALTH_DEFAULT='http://127.0.0.1:8000/api/health'
ALL_CONFIGS=['B0','B1','B2','A1','A2','A3','A4']


def pct(x): return None if x is None else float(x)


def calc_metrics(g):
    valid=g[g['pred_label'].isin(['BENIGN','SCAM'])].copy()
    if len(valid)==0:
        return {'n':len(g),'valid_n':0}
    y=valid['true_label'].astype(int).values
    p=valid['pred_label'].map({'BENIGN':0,'SCAM':1}).astype(int).values
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    lat=valid['total_ms'].dropna().astype(float).values
    return {
        'n':int(len(g)),'valid_n':int(len(valid)),
        'valid_rate':float(len(valid)/len(g)),
        'accuracy':float(accuracy_score(y,p)),
        'precision':float(precision_score(y,p,zero_division=0)),
        'recall':float(recall_score(y,p,zero_division=0)),
        'f1':float(f1_score(y,p,zero_division=0)),
        'specificity':float(tn/(tn+fp)) if tn+fp else None,
        'fpr':float(fp/(fp+tn)) if fp+tn else None,
        'fnr':float(fn/(fn+tp)) if fn+tp else None,
        'balanced_accuracy':float(balanced_accuracy_score(y,p)),
        'mcc':float(matthews_corrcoef(y,p)),
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
        'qwen_invocation_rate':float(g['qwen_invoked'].fillna(False).mean()),
        'qwen_parser_valid_rate_when_invoked':(
            float(g[g.qwen_invoked==True]['qwen_parser_valid'].fillna(False).mean())
            if (g.qwen_invoked==True).any() else None
        ),
        'latency_mean_ms':float(lat.mean()) if len(lat) else None,
        'latency_median_ms':float(np.median(lat)) if len(lat) else None,
        'latency_p95_ms':float(np.percentile(lat,95)) if len(lat) else None,
        'latency_p99_ms':float(np.percentile(lat,99)) if len(lat) else None,
        'lr_mean_ms':float(g['lr_ms'].dropna().mean()) if g['lr_ms'].notna().any() else None,
        'rules_mean_ms':float(g['rules_ms'].dropna().mean()) if g['rules_ms'].notna().any() else None,
        'rag_mean_ms':float(g['rag_ms'].dropna().mean()) if g['rag_ms'].notna().any() else None,
        'qwen_mean_ms':float(g['qwen_ms'].dropna().mean()) if g['qwen_ms'].notna().any() else None,
    }


def post(session,endpoint,payload,timeout,retries=3):
    last=None
    for attempt in range(1,retries+1):
        t0=time.perf_counter()
        try:
            r=session.post(endpoint,json=payload,timeout=timeout)
            wall_ms=(time.perf_counter()-t0)*1000
            if r.status_code==200:
                return True,r.json(),wall_ms,attempt,''
            last=f'HTTP {r.status_code}: {r.text[:500]}'
        except Exception as e:
            last=repr(e)
        time.sleep(min(5,attempt*2))
    return False,None,None,retries,last


def run_independent(endpoint,health_url,configs,timeout):
    health=requests.get(health_url,timeout=10).json()
    if health.get('api_version')!='research_v1':
        raise SystemExit(f'Wrong backend: {health}')
    if any(c in {'A2','A4'} for c in configs) and not health.get('rag_ready'):
        raise SystemExit('RAG is not ready. Run 03_rag_benchmark.py and 04_build_chroma_research.py first.')

    df=pd.read_csv(DATA/'independent_test_v1.csv')
    session=requests.Session(); rows=[]
    total=len(df)*len(configs); k=0
    for config in configs:
        for _,r in df.iterrows():
            k+=1
            ok,data,wall,attempts,error=post(session,endpoint,{
                'query':r.text,'config':config,'decision_mode':'binary'
            },timeout)
            if not ok:
                rows.append({
                    'sample_id':r.sample_id,'text':r.text,'true_label':int(r.label),
                    'scenario_id':r.scenario_id,'config':config,'pred_label':'ERROR',
                    'request_ok':False,'error':error,'attempts':attempts,'wall_ms':wall,
                })
            else:
                t=data.get('timings_ms') or {}
                rows.append({
                    'sample_id':r.sample_id,'text':r.text,'true_label':int(r.label),
                    'scenario_id':r.scenario_id,'config':config,'pred_label':data.get('label'),
                    'request_ok':True,'error':data.get('error_type','NONE'),'attempts':attempts,
                    'wall_ms':wall,'lr_prob':data.get('lr_scam_probability'),
                    'rule_hits':json.dumps(data.get('rule_hits',[]),ensure_ascii=False),
                    'strong_rule':data.get('strong_rule'),
                    'qwen_invoked':data.get('qwen_invoked'),
                    'qwen_parser_valid':data.get('qwen_parser_valid'),
                    'retrieved_count':len(data.get('retrieved') or []),
                    'decision_path':data.get('decision_path'),
                    'lr_ms':t.get('lr'),'rules_ms':t.get('rules'),'rag_ms':t.get('rag'),
                    'qwen_ms':t.get('qwen'),'total_ms':t.get('total'),
                })
            print(f'\rIndependent {k}/{total} | {config}',end='')
    print()
    out=pd.DataFrame(rows)
    out.to_csv(RESULTS/'ablation_predictions.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for config in configs:
        m=calc_metrics(out[out.config==config])
        m['config']=config; summary.append(m)
    sm=pd.DataFrame(summary)
    sm.to_csv(RESULTS/'ablation_summary.csv',index=False,encoding='utf-8-sig')
    return out,sm,health


def run_ood(endpoint,timeout):
    df=pd.read_csv(DATA/'ood_gray_v1.csv')
    session=requests.Session(); rows=[]
    for i,r in df.iterrows():
        ok,data,wall,attempts,error=post(session,endpoint,{
            'query':r.text,'config':'A4','decision_mode':'selective'
        },timeout)
        if not ok:
            pred='ERROR'; path='ERROR'; q=False; qv=False; prob=None; t={}
        else:
            pred=data.get('label'); path=data.get('decision_path'); q=data.get('qwen_invoked'); qv=data.get('qwen_parser_valid'); prob=data.get('lr_scam_probability'); t=data.get('timings_ms') or {}
        rows.append({
            'sample_id':r.sample_id,'text':r.text,'expected_action':r.expected_action,
            'scenario_id':r.scenario_id,'scenario_type':r.scenario_type,
            'pred_action':pred,'correct_action':pred==r.expected_action,
            'lr_prob':prob,'qwen_invoked':q,'qwen_parser_valid':qv,
            'decision_path':path,'total_ms':t.get('total'),'error':error if not ok else data.get('error_type','NONE'),
        })
        print(f'\rOOD/selective {i+1}/{len(df)}',end='')
    print()
    out=pd.DataFrame(rows)
    out.to_csv(RESULTS/'ood_gray_predictions.csv',index=False,encoding='utf-8-sig')
    summary={
        'n':len(out),
        'action_accuracy':float(out.correct_action.mean()),
        'scam_catch_rate':float((out[out.expected_action=='SCAM'].pred_action=='SCAM').mean()),
        'benign_accept_rate':float((out[out.expected_action=='BENIGN'].pred_action=='BENIGN').mean()),
        'gray_abstention_rate':float((out[out.expected_action=='UNCERTAIN'].pred_action=='UNCERTAIN').mean()),
        'overall_uncertain_rate':float((out.pred_action=='UNCERTAIN').mean()),
        'qwen_invocation_rate':float(out.qwen_invoked.fillna(False).mean()),
        'mean_latency_ms':float(out.total_ms.dropna().mean()) if out.total_ms.notna().any() else None,
    }
    (RESULTS/'ood_gray_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return out,summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--endpoint',default=ENDPOINT_DEFAULT)
    ap.add_argument('--health',default=HEALTH_DEFAULT)
    ap.add_argument('--configs',default=','.join(ALL_CONFIGS))
    ap.add_argument('--timeout',type=int,default=300)
    ap.add_argument('--skip-ood',action='store_true')
    a=ap.parse_args(); RESULTS.mkdir(exist_ok=True)
    configs=[x.strip() for x in a.configs.split(',') if x.strip()]
    unknown=set(configs)-set(ALL_CONFIGS)
    if unknown: raise SystemExit(f'Unknown configs: {unknown}')
    _,sm,health=run_independent(a.endpoint,a.health,configs,a.timeout)
    print('\nABLATION SUMMARY\n',sm.to_string(index=False))
    if not a.skip_ood and 'A4' in configs:
        _,ood=run_ood(a.endpoint,a.timeout)
        print('\nOOD/GRAY SUMMARY\n',json.dumps(ood,ensure_ascii=False,indent=2))
    (RESULTS/'run_environment.json').write_text(json.dumps(health,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
