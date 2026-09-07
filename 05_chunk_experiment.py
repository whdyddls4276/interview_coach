# -*- coding: utf-8 -*-
"""
경험 C 소재용 — 실제 청킹(chunk_size) 비교 실험
- 01_rag_build.ipynb의 chunk_text 로직을 그대로 재사용
- chunk_size 200 / 300(기존 하드코딩값) / 500 세 가지를 비교
- 각 설정으로 임베딩(paraphrase-multilingual-MiniLM-L12-v2) 후,
  테스트 쿼리에 대해 top-3 검색 → 검색된 컨텍스트로 Gemini(파인튜닝 없는 일반 모델)에게
  RAG 기반 답변만 생성시켜 결과를 비교한다.
"""
import io, sys, os, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import numpy as np

BASE = Path(r'C:\Users\82105\OneDrive\바탕 화면\프로젝트3(면접)')
TXT_DIRS = [
    BASE / '샘플영상' / '이형 유튜브 영상' / '원본' / 'output_txt',
    BASE / '샘플영상' / '이형 유튜브 영상' / '추가' / 'output_txt',
    BASE / '샘플영상' / '이형 유튜브 영상' / '피드백편집본' / 'output_txt',
]

def load_txt_files(dirs):
    docs = []
    for d in dirs:
        if not d.exists():
            print(f'[경고] 없음: {d}')
            continue
        for f in sorted(d.glob('*.txt')):
            text = f.read_text(encoding='utf-8', errors='ignore').strip()
            if text:
                docs.append({'source': f.stem, 'text': text})
    return docs

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 10]

def chunk_text(text, source, chunk_size=300, overlap=50):
    sentences = split_sentences(text)
    chunks = []
    buf, buf_len = [], 0
    for sent in sentences:
        buf.append(sent)
        buf_len += len(sent)
        if buf_len >= chunk_size:
            chunks.append({'text': ' '.join(buf), 'source': source})
            overlap_sents = []
            ol = 0
            for s in reversed(buf):
                if ol + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                ol += len(s)
            buf, buf_len = overlap_sents, ol
    if buf:
        chunks.append({'text': ' '.join(buf), 'source': source})
    return chunks

print('=== 1. 소스 문서 로드 ===')
docs = load_txt_files(TXT_DIRS)
print(f'총 {len(docs)}개 문서 로드 (원본/추가/피드백편집본 합계, 현재 폴더 기준)')

CONFIGS = [
    {'name': 'chunk_size=200, overlap=50', 'chunk_size': 200, 'overlap': 50},
    {'name': 'chunk_size=300, overlap=50 (기존 하드코딩값)', 'chunk_size': 300, 'overlap': 50},
    {'name': 'chunk_size=500, overlap=50', 'chunk_size': 500, 'overlap': 50},
    {'name': 'chunk_size=300, overlap=0 (overlap 없음)', 'chunk_size': 300, 'overlap': 0},
]

print('\n=== 2. 설정별 청킹 결과 (구조 비교) ===')
chunked_by_config = {}
for cfg in CONFIGS:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc['text'], doc['source'], cfg['chunk_size'], cfg['overlap']))
    lens = [len(c['text']) for c in all_chunks]
    chunked_by_config[cfg['name']] = all_chunks
    print(f"- {cfg['name']:<38} 청크수={len(all_chunks):>4}  평균길이={sum(lens)//len(lens):>4}자  "
          f"최소={min(lens):>4}  최대={max(lens):>4}")

print('\n=== 3. 임베딩 모델 로드 (paraphrase-multilingual-MiniLM-L12-v2) ===')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print('로드 완료')

def embed(texts):
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

def top_k(query_vec, chunk_vecs, chunks, k=3):
    sims = chunk_vecs @ query_vec
    idx = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in idx]

TEST_QUERIES = [
    '면접에서 두괄식으로 답변하는 방법이 궁금해요',
    '면접관에게 좋은 인상을 주려면 어떻게 해야 하나요',
]

print('\n=== 4. 설정별 검색 결과 비교 (top-3) ===')
retrieval_results = {}
for cfg_name, chunks in chunked_by_config.items():
    print(f'\n--- {cfg_name} ---')
    texts = [c['text'] for c in chunks]
    chunk_vecs = embed(texts)
    retrieval_results[cfg_name] = {}
    for q in TEST_QUERIES:
        qvec = embed([q])[0]
        results = top_k(qvec, chunk_vecs, chunks, k=3)
        retrieval_results[cfg_name][q] = results
        print(f'  질의: "{q}"')
        for c, score in results:
            preview = c['text'][:80].replace('\n', ' ')
            print(f'    sim={score:.3f}  [{c["source"][:20]}]  {preview}...')

# 결과 저장 (다음 단계: Gemini 답변 생성에서 재사용)
out = {
    'doc_count': len(docs),
    'configs': {name: [{'text': c['text'], 'source': c['source']} for c in chs]
                for name, chs in chunked_by_config.items()},
    'retrieval': {
        cfg: {q: [{'text': c['text'], 'source': c['source'], 'score': s} for c, s in res]
              for q, res in qs.items()}
        for cfg, qs in retrieval_results.items()
    }
}
OUT_PATH = Path(r'C:\Users\82105\AppData\Local\Temp\claude\c--python\e53b1ec6-7e8d-470a-8f40-e105c3bb1978\scratchpad\chunk_experiment_result.json')
OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'\n결과 저장: {OUT_PATH}')
