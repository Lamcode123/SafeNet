from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

import joblib
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/'freeze'
RAG_DB=ROOT/'chroma_research_v1'

MODEL_PATH=FREEZE/'safenet_lr_research_v1.joblib'
CAL_PATH=FREEZE/'lr_calibration_v1.json'
RULES_PATH=FREEZE/'rules_v1.json'
RAG_SEL_PATH=FREEZE/'rag_selection_v1.json'
COLLECTION='safenet_rag_research_v1'

OLLAMA_URL=os.getenv('OLLAMA_URL','http://127.0.0.1:11434/api/generate')
OLLAMA_MODEL=os.getenv('OLLAMA_MODEL','qwen2.5:1.5b')

CONFIGS={'B0','B1','B2','A1','A2','A3','A4'}

lr_model=joblib.load(MODEL_PATH)
cal=json.loads(CAL_PATH.read_text(encoding='utf-8'))
LOW=float(cal['benign_threshold']); HIGH=float(cal['scam_threshold'])
rules_doc=json.loads(RULES_PATH.read_text(encoding='utf-8'))

rag_collection=None
rag_max_distance=None
rag_model_name=None
rag_error=None
if RAG_SEL_PATH.exists() and RAG_DB.exists():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        sel=json.loads(RAG_SEL_PATH.read_text(encoding='utf-8'))
        rag_model_name=sel['selected_embedding_model']
        rag_max_distance=float(sel['distance_threshold_for_chroma_cosine'])
        ef=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=rag_model_name)
        client=chromadb.PersistentClient(path=str(RAG_DB))
        rag_collection=client.get_collection(COLLECTION,embedding_function=ef)
    except Exception as e:
        rag_error=repr(e)

app=FastAPI(title='SafeNet Final Research Server',version='research_v1')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])


class ResearchRequest(BaseModel):
    query:str
    config:Literal['B0','B1','B2','A1','A2','A3','A4']='A4'
    decision_mode:Literal['binary','selective']='binary'


class ResearchResponse(BaseModel):
    config:str
    decision_mode:str
    label:str
    lr_scam_probability:Optional[float]=None
    rule_hits:list[str]=Field(default_factory=list)
    strong_rule:bool=False
    qwen_invoked:bool=False
    qwen_parser_valid:Optional[bool]=None
    qwen_reason:Optional[str]=None
    retrieved:list[dict]=Field(default_factory=list)
    decision_path:str
    timings_ms:dict=Field(default_factory=dict)
    error_type:str='NONE'


def analyze_rules(text:str):
    q=text.lower()
    hits=[]
    for rule in rules_doc['rules']:
        ok=True
        for group in rule['groups']:
            if not any(str(term).lower() in q for term in group):
                ok=False; break
        if ok:
            hits.append(rule)
    return hits


def retrieve(text:str,top_k=3):
    if rag_collection is None:
        raise RuntimeError(f'RAG not ready: {rag_error or "run 03 and 04 first"}')
    res=rag_collection.query(query_texts=[text],n_results=top_k,include=['documents','metadatas','distances'])
    docs=(res.get('documents') or [[]])[0]
    metas=(res.get('metadatas') or [[]])[0]
    dists=(res.get('distances') or [[]])[0]
    out=[]
    for doc,meta,dist in zip(docs,metas,dists):
        dist=float(dist)
        if dist <= rag_max_distance:
            out.append({'document':str(doc),'metadata':meta or {},'distance':dist,'similarity':1.0-dist})
    return out


SCHEMA_BINARY={
    'type':'object',
    'properties':{
        'label':{'type':'string','enum':['BENIGN','SCAM']},
        'reason':{'type':'string'},
    },
    'required':['label','reason'],
    'additionalProperties':False,
}
SCHEMA_SELECTIVE={
    'type':'object',
    'properties':{
        'label':{'type':'string','enum':['BENIGN','SCAM','UNCERTAIN']},
        'reason':{'type':'string'},
    },
    'required':['label','reason'],
    'additionalProperties':False,
}


def qwen_call(text:str,mode:str='neutral',rag_items=None,selective=False):
    rag_items=rag_items or []
    if mode=='neutral':
        instructions='''Classify the Vietnamese message as BENIGN or SCAM. Use only evidence in the message. Do not invent facts.'''
    else:
        instructions='''Classify the Vietnamese message. A SCAM tries to obtain money, credentials, OTPs, account access, or compliance through deception, impersonation, coercion, fake rewards, advance fees, or suspicious links. A BENIGN message may legitimately discuss money, banks, work, shopping, family, or links without deceptive solicitation. Do not treat the mere presence of money/bank/link words as proof of scam.\nBalanced examples:\nBENIGN: "Mẹ đã chuyển tiền học như đã thống nhất, con kiểm tra tài khoản nhé."\nBENIGN: "Phòng nhân sự gửi bảng lương trên cổng nội bộ; không yêu cầu OTP."\nSCAM: "Tôi là công an, chuyển tiền vào tài khoản tạm giữ để chứng minh vô tội."\nSCAM: "Bạn trúng thưởng, đóng phí trước rồi mới nhận giải."'''
    if selective:
        instructions += '\nIf the evidence is genuinely insufficient or conflicting, use UNCERTAIN.'

    context='\n'.join(
        f"[{x['metadata'].get('scenario_id','KB')}] {x['document']}" for x in rag_items
    ) or 'NO_RETRIEVED_CONTEXT'
    prompt=f'''You are the bounded advisory component of SafeNet.\n{instructions}\n\nRetrieved knowledge is reference-only and is NOT evidence that the user message contains those facts.\nRetrieved knowledge:\n{context}\n\nUSER MESSAGE:\n<<<{text}>>>\n\nReturn JSON only.'''
    payload={
        'model':OLLAMA_MODEL,'prompt':prompt,'stream':False,
        'format':SCHEMA_SELECTIVE if selective else SCHEMA_BINARY,
        'options':{'temperature':0.0,'num_predict':100},
    }
    try:
        r=requests.post(OLLAMA_URL,json=payload,timeout=300)
        r.raise_for_status()
        raw=r.json().get('response','').strip()
        obj=json.loads(raw)
        label=str(obj.get('label','')).upper().strip()
        allowed={'BENIGN','SCAM','UNCERTAIN'} if selective else {'BENIGN','SCAM'}
        if label not in allowed:
            raise ValueError(f'invalid label {label!r}')
        return label,str(obj.get('reason','')).strip(),True,'NONE'
    except Exception as e:
        return None,repr(e),False,'QWEN_ERROR'


def lr_predict(text):
    return float(lr_model.predict_proba([text])[0][1])


def verify(req:ResearchRequest):
    text=req.query.strip()
    if not text:
        raise ValueError('empty query')
    config=req.config; selective=req.decision_mode=='selective'
    t0=time.perf_counter(); timings={}
    p=None; hits=[]; strong=False; rag_items=[]; q_label=None; q_reason=None; q_valid=None; err='NONE'

    # B1: Qwen neutral only.
    if config=='B1':
        tq=time.perf_counter(); q_label,q_reason,q_valid,err=qwen_call(text,'neutral',[],selective=False); timings['qwen']=1000*(time.perf_counter()-tq)
        label=q_label or 'ERROR'; path='QWEN_ZERO_SHOT'

    # B2: Qwen balanced few-shot only.
    elif config=='B2':
        tq=time.perf_counter(); q_label,q_reason,q_valid,err=qwen_call(text,'balanced',[],selective=False); timings['qwen']=1000*(time.perf_counter()-tq)
        label=q_label or 'ERROR'; path='QWEN_BALANCED'

    # A2: RAG + Qwen on every sample.
    elif config=='A2':
        tr=time.perf_counter(); rag_items=retrieve(text); timings['rag']=1000*(time.perf_counter()-tr)
        tq=time.perf_counter(); q_label,q_reason,q_valid,err=qwen_call(text,'balanced',rag_items,selective=False); timings['qwen']=1000*(time.perf_counter()-tq)
        label=q_label or 'ERROR'; path='RAG_QWEN'

    else:
        tl=time.perf_counter(); p=lr_predict(text); timings['lr']=1000*(time.perf_counter()-tl)

        if config=='B0':
            label='SCAM' if p>=0.5 else 'BENIGN'; path='LR_ONLY'
        else:
            trule=time.perf_counter(); hits=analyze_rules(text); timings['rules']=1000*(time.perf_counter()-trule)
            strong=any(bool(x.get('strong')) for x in hits)

            if config=='A1':
                if strong and p>=0.40:
                    label='SCAM'; path='LR_RULE_OVERRIDE'
                else:
                    label='SCAM' if p>=0.5 else 'BENIGN'; path='LR_RULE_FALLBACK'
            else:
                # A3/A4: deterministic high-confidence path.
                if p>=HIGH:
                    label='SCAM'; path='LR_HIGH_SCAM'
                elif p<=LOW and not strong:
                    label='BENIGN'; path='LR_HIGH_BENIGN'
                elif strong and p>=0.40:
                    label='SCAM'; path='RULE_ESCALATION'
                else:
                    if config=='A4':
                        tr=time.perf_counter(); rag_items=retrieve(text); timings['rag']=1000*(time.perf_counter()-tr)
                    tq=time.perf_counter(); q_label,q_reason,q_valid,err=qwen_call(text,'balanced',rag_items,selective=selective); timings['qwen']=1000*(time.perf_counter()-tq)
                    if not q_valid:
                        if selective:
                            label='UNCERTAIN'; path='AMBIGUOUS_QWEN_ERROR_HOLD'
                        else:
                            label='SCAM' if p>=0.5 else 'BENIGN'; path='AMBIGUOUS_QWEN_ERROR_LR_FALLBACK'
                    elif not selective:
                        # Bounded authority: Qwen decides only inside the calibrated ambiguous band.
                        label=q_label; path='AMBIGUOUS_QWEN_DECISION'
                    else:
                        lr_dir='SCAM' if p>=0.5 else 'BENIGN'
                        if q_label==lr_dir:
                            label=q_label; path='AMBIGUOUS_QWEN_CONFIRM'
                        else:
                            label='UNCERTAIN'; path='AMBIGUOUS_SAFETY_HOLD'

    timings['total']=1000*(time.perf_counter()-t0)
    return ResearchResponse(
        config=config,decision_mode=req.decision_mode,label=label,
        lr_scam_probability=p,rule_hits=[x['id'] for x in hits],strong_rule=strong,
        qwen_invoked=q_valid is not None,qwen_parser_valid=q_valid,qwen_reason=q_reason,
        retrieved=rag_items,decision_path=path,timings_ms=timings,error_type=err,
    )


@app.post('/api/research/verify',response_model=ResearchResponse)
def api_verify(req:ResearchRequest):
    return verify(req)


@app.get('/api/health')
def health():
    return {
        'status':'ok','api_version':'research_v1','server_pid':os.getpid(),
        'lr_ready':True,'rules_version':rules_doc['version'],
        'gate':{'low':LOW,'high':HIGH},
        'rag_ready':rag_collection is not None,'rag_model':rag_model_name,
        'rag_max_distance':rag_max_distance,'rag_error':rag_error,
        'qwen_model':OLLAMA_MODEL,
    }


if __name__=='__main__':
    import uvicorn
    print('SafeNet Research Server v1')
    print(json.dumps(health(),ensure_ascii=False,indent=2))
    uvicorn.run(app,host='127.0.0.1',port=8000)
