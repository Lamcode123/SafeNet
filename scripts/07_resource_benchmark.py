from __future__ import annotations

import argparse
import json
import statistics
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import requests

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; RESULTS=ROOT/'results'


def pctl(xs,q): return float(np.percentile(xs,q)) if len(xs) else None


def collect_targets(server_pid):
    procs=[]
    try: procs.append(psutil.Process(server_pid))
    except Exception: pass
    for p in psutil.process_iter(['pid','name']):
        try:
            name=(p.info.get('name') or '').lower()
            if 'ollama' in name and p.pid != server_pid:
                procs.append(p)
        except Exception: pass
    return procs


def sample_resources(stop,server_pid,store,interval=0.10):
    psutil.cpu_percent(None)
    while not stop.is_set():
        server_rss=0; ollama_rss=0
        for p in collect_targets(server_pid):
            try:
                rss=p.memory_info().rss/1024/1024
                name=p.name().lower()
                if p.pid==server_pid: server_rss += rss
                elif 'ollama' in name: ollama_rss += rss
            except Exception: pass
        store.append({
            'system_cpu_pct':psutil.cpu_percent(None),
            'system_mem_pct':psutil.virtual_memory().percent,
            'server_rss_mb':server_rss,
            'ollama_rss_mb':ollama_rss,
        })
        stop.wait(interval)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--endpoint',default='http://127.0.0.1:8000/api/research/verify')
    ap.add_argument('--health',default='http://127.0.0.1:8000/api/health')
    ap.add_argument('--configs',default='B0,B1,A4')
    ap.add_argument('--samples',type=int,default=30)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--timeout',type=int,default=300)
    a=ap.parse_args(); RESULTS.mkdir(exist_ok=True)

    health=requests.get(a.health,timeout=10).json(); server_pid=int(health['server_pid'])
    env={
        'platform':platform.platform(),
        'machine':platform.machine(),
        'processor':platform.processor(),
        'python_version':sys.version,
        'logical_cpu_count':psutil.cpu_count(logical=True),
        'physical_cpu_count':psutil.cpu_count(logical=False),
        'total_ram_gb':round(psutil.virtual_memory().total/(1024**3),3),
        'server_health':health,
    }
    (RESULTS/'resource_environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
    df=pd.read_csv(DATA/'independent_test_v1.csv')
    # balanced, deterministic subset: first half benign / first half scam after sample_id sort
    n_each=a.samples//2
    subset=pd.concat([
        df[df.label==0].sort_values('sample_id').head(n_each),
        df[df.label==1].sort_values('sample_id').head(a.samples-n_each),
    ]).sample(frac=1,random_state=20261002).reset_index(drop=True)
    configs=[x.strip() for x in a.configs.split(',') if x.strip()]
    rows=[]; session=requests.Session(); total=len(configs)*len(subset)*a.repeats; k=0

    for config in configs:
        for rep in range(1,a.repeats+1):
            for _,r in subset.iterrows():
                k+=1; samples=[]; stop=threading.Event()
                th=threading.Thread(target=sample_resources,args=(stop,server_pid,samples),daemon=True); th.start()
                t0=time.perf_counter(); err=''; data=None
                try:
                    resp=session.post(a.endpoint,json={'query':r.text,'config':config,'decision_mode':'binary'},timeout=a.timeout)
                    resp.raise_for_status(); data=resp.json()
                except Exception as e: err=repr(e)
                wall=(time.perf_counter()-t0)*1000
                stop.set(); th.join(timeout=2)
                # final resource sample if request too fast
                if not samples:
                    samples.append({'system_cpu_pct':psutil.cpu_percent(None),'system_mem_pct':psutil.virtual_memory().percent,'server_rss_mb':0,'ollama_rss_mb':0})
                rows.append({
                    'config':config,'repeat':rep,'sample_id':r.sample_id,'true_label':int(r.label),
                    'pred_label':data.get('label') if data else 'ERROR','wall_ms':wall,
                    'server_total_ms':(data.get('timings_ms') or {}).get('total') if data else None,
                    'qwen_invoked':data.get('qwen_invoked') if data else None,
                    'peak_system_cpu_pct':max(x['system_cpu_pct'] for x in samples),
                    'peak_system_mem_pct':max(x['system_mem_pct'] for x in samples),
                    'peak_server_rss_mb':max(x['server_rss_mb'] for x in samples),
                    'peak_ollama_rss_mb':max(x['ollama_rss_mb'] for x in samples),
                    'error':err,
                })
                print(f'\rResource {k}/{total} | {config} rep={rep}',end='')
    print()
    raw=pd.DataFrame(rows); raw.to_csv(RESULTS/'resource_raw.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for config,g in raw.groupby('config'):
        lat=g.wall_ms.dropna().values
        summary.append({
            'config':config,'n':len(g),
            'latency_mean_ms':float(np.mean(lat)),'latency_median_ms':float(np.median(lat)),
            'latency_p95_ms':pctl(lat,95),'latency_p99_ms':pctl(lat,99),
            'peak_system_cpu_pct':float(g.peak_system_cpu_pct.max()),
            'mean_peak_system_cpu_pct':float(g.peak_system_cpu_pct.mean()),
            'peak_server_rss_mb':float(g.peak_server_rss_mb.max()),
            'peak_ollama_rss_mb':float(g.peak_ollama_rss_mb.max()),
            'qwen_invocation_rate':float(g.qwen_invoked.fillna(False).mean()),
            'error_rate':float((g.error.fillna('')!='').mean()),
        })
    sm=pd.DataFrame(summary); sm.to_csv(RESULTS/'resource_summary.csv',index=False,encoding='utf-8-sig')
    print(sm.to_string(index=False))

if __name__=='__main__': main()
