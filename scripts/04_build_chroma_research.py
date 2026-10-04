from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; FREEZE=ROOT/'freeze'
DB=ROOT/'chroma_research_v1'
COLLECTION='safenet_rag_research_v1'


def main():
    import chromadb
    from chromadb.utils import embedding_functions

    sel=json.loads((FREEZE/'rag_selection_v1.json').read_text(encoding='utf-8'))
    model_name=sel['selected_embedding_model']
    kb=pd.read_csv(DATA/'rag_knowledge_v1.csv')

    client=chromadb.PersistentClient(path=str(DB))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    ef=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    col=client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={'hnsw:space':'cosine','version':'research_v1'},
    )
    documents=(kb.title.fillna('')+'. '+kb.content.fillna('')).tolist()
    metadatas=[{
        'scenario_id':r.scenario_id,'title':r.title,'source_url':r.source_url
    } for r in kb.itertuples(index=False)]
    ids=[f'kb_{x}' for x in kb.scenario_id]
    col.add(ids=ids,documents=documents,metadatas=metadatas)
    print(f'Built {COLLECTION}: {col.count()} docs | model={model_name} | db={DB}')

if __name__=='__main__': main()
