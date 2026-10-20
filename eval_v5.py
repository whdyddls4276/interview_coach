"""
eval_v5.py — v5 = v4에서 검색 임베딩만 MiniLM → gemini-embedding-001로 교체

embed_compare.py에서 v4 검색 질문 31개의 근거 관련도가 MiniLM 40% → gemini-embedding-001 60%로 올랐다
(같은 판정 안에서 비교). 검색이 좋아지면 피드백(코치 지적 재현율)도 좋아지는지 본다.
질의(v3_query)·예시(v3_shots)·프롬프트(SYSTEM_PROMPT_V1)는 v4와 같고, 근거 4개만 다르다.
  v5 vs v4 = 임베딩(검색 근거) 효과
결과: eval_results_v2/full/generation.json 에 v5, v5_ctx 추가
"""
import json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from eval_v2 import GEN_MODEL, OUT, PROJ, load_pairs
from embed_compare import EMB_DIR, Ranker, load_corpus

FULL = OUT / 'full'
EMB = 'gemini-embedding-001'
lock = threading.Lock()


def main():
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from sklearn.feature_extraction.text import TfidfVectorizer
    from embed_gemini import gemini_embed
    from prompts_v2 import SYSTEM_PROMPT_V1

    load_dotenv(PROJ / '.env')
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY', ''))
    pairs = {p['id']: p for p in load_pairs()}
    path = FULL / 'generation.json'
    gen = json.load(open(path, encoding='utf-8'))

    ids, docs, metas, _ = load_corpus()
    assert json.load(open(EMB_DIR / f'{EMB}_docs_ids.json', encoding='utf-8')) == ids, '저장된 임베딩과 DB 순서가 다름'
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)).fit(docs)
    ranker = Ranker(np.load(EMB_DIR / f'{EMB}_docs.npy'), tfidf, tfidf.transform(docs))
    qcache = json.load(open(EMB_DIR / f'{EMB}_queries.json', encoding='utf-8'))

    def qvec(q):
        if q not in qcache:
            with lock:
                qcache[q] = gemini_embed(client, EMB, [q], 'RETRIEVAL_QUERY')[0].tolist()
        return np.asarray(qcache[q], dtype='float32')

    def run(k):
        g = gen[k]
        if g.get('v5') or not g.get('v3_query'):
            return
        ctx_docs = [docs[i] for i in ranker.top(g['v3_query'], qvec(g['v3_query']), 'hybrid', 4)]
        c = []
        for sid in g['v3_shots']:
            s = pairs[sid]
            c += [types.Content(role='user', parts=[types.Part.from_text(text=f'Q.\n{s["interview"]}')]),
                  types.Content(role='model', parts=[types.Part.from_text(text=f'A.\n{s["feedback"]}')])]
        ctx = '\n\n'.join(f'[참고{i+1}] {d}' for i, d in enumerate(ctx_docs))
        c.append(types.Content(role='user', parts=[types.Part.from_text(
            text=f'Q.\n{g["interview"]}\n\n--- 참고 자료 (면접왕 이형 강의) ---\n{ctx}')]))
        for t in range(5):
            try:
                r = client.models.generate_content(model=GEN_MODEL, contents=c, config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_V1, temperature=0.4, max_output_tokens=4096))
                if r.text:
                    g['v5'] = re.sub(r'^\s*A\.\s*', '', r.text.strip())
                    g['v5_ctx'] = ctx_docs
                    break
            except Exception as e:
                print('   재시도:', str(e)[:60]); time.sleep(60 if '429' in str(e) else 20)
        with lock:
            json.dump(gen, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            done = sum(1 for v in gen.values() if v.get('v5'))
        same = sum(d in g['v4_ctx'] for d in ctx_docs)
        print(f'  {k:16s} v5 {len(g.get("v5", ""))}자 | v4와 같은 근거 {same}/4 | 완료 {done}/{len(gen)}')

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(run, list(gen)))
    print('완료')


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    main()
