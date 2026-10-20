"""
embed_compare.py — 임베딩 모델 비교 (면접 강의 DB 567청크)

지금 쓰는 MiniLM(paraphrase-multilingual-MiniLM-L12-v2)은 128토큰까지만 읽는다 → 청크 89%가 잘려 있다
(청크 길이 중앙값 165토큰, 잘린 청크는 평균 뒤 23%를 못 읽음). Gemini 임베딩 API로 바꿔 같은 기준으로 비교한다.
  - gemini-embedding-001: 2,048토큰 / gemini-embedding-2: 8,192토큰, 둘 다 3,072차원
  - 문서는 RETRIEVAL_DOCUMENT, 질문은 RETRIEVAL_QUERY로 임베딩

평가
1) 주제 질문 14개 (eval_v2.QUESTIONS): 정답 강의 Hit@1 · Hit@3 · MRR@5 — 임베딩만 / 키워드 결합(RRF, 서비스 방식)
2) v4 실제 검색 질문 31개 (generation.json의 v3_query): 키워드 결합 상위 4개의 근거 관련도
   세 모델이 가져온 문단을 쌍마다 모아 Claude가 한 번에 0/1 판정 → 같은 문단은 세 모델에 같은 판정 적용
   (판정의 흔들림이 모델 비교에 섞이지 않게)

기존 chroma_db는 복사본만 읽는다. 임베딩은 eval_results_v2/embed/에 저장해 다시 실행하면 재사용.
결과: eval_results_v2/embed_compare.json
"""
import json, os, random, re, shutil, tempfile

import numpy as np

from eval_v2 import BASE, OUT, PROJ, QUESTIONS, score_ranking
from embed_gemini import gemini_embed
from rag_v2 import CHROMA_DIR, COLLECTION, EMBED_MODEL

EMB_DIR = OUT / 'embed'
MODELS = ('minilm', 'gemini-embedding-001', 'gemini-embedding-2')
K_RRF = 60


def load_corpus():
    import chromadb
    tmp = os.path.join(tempfile.mkdtemp(), 'db')
    shutil.copytree(CHROMA_DIR, tmp)
    r = chromadb.PersistentClient(path=tmp).get_collection(COLLECTION).get(
        include=['documents', 'metadatas', 'embeddings'], limit=100000)
    return r['ids'], r['documents'], r['metadatas'], np.asarray(r['embeddings'], dtype='float32')


class Ranker:
    """rag_v2.HybridRetriever와 같은 계산 (임베딩만 바꿔 끼울 수 있게)"""

    def __init__(self, emb, tfidf, tx):
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.mu = self.emb.mean(axis=0)
        self.tfidf, self.tx = tfidf, tx

    def scores(self, query, qv, mode):
        s_emb = (self.emb - self.mu) @ (qv - self.mu)          # 서비스 방식: 평균 벡터를 뺀 유사도
        if mode == 'emb':
            return s_emb
        s_kw = (self.tx @ self.tfidf.transform([query]).T).toarray().ravel()
        r_kw = np.argsort(np.argsort(-s_kw)); r_emb = np.argsort(np.argsort(-s_emb))
        return 1 / (K_RRF + r_kw) + 1 / (K_RRF + r_emb)

    def top(self, query, qv, mode, k):
        return list(np.argsort(-self.scores(query, qv, mode))[:k])


def main():
    import anthropic
    from dotenv import dotenv_values, load_dotenv
    from google import genai
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from judge_v2 import JUDGE_MODEL, PRICE_IN, PRICE_OUT, SYSTEM

    EMB_DIR.mkdir(exist_ok=True)
    load_dotenv(PROJ / '.env')
    gclient = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY', ''))
    ids, docs, metas, mini = load_corpus()
    sources = [m['source'] for m in metas]
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)).fit(docs)
    tx = tfidf.transform(docs)

    gen = json.load(open(OUT / 'full' / 'generation.json', encoding='utf-8'))
    gold = json.load(open(OUT / 'full' / 'gold.json', encoding='utf-8'))['items']
    keys = [k for k in gold if gen.get(k, {}).get('v3_query')]
    queries = list(QUESTIONS) + [gen[k]['v3_query'] for k in keys]

    # ── 임베딩 (저장된 게 있으면 재사용) ──
    doc_emb, q_emb = {'minilm': mini}, {}
    st = SentenceTransformer(EMBED_MODEL)
    q_emb['minilm'] = st.encode(queries, normalize_embeddings=True)
    for m in MODELS[1:]:
        f, fid = EMB_DIR / f'{m}_docs.npy', EMB_DIR / f'{m}_docs_ids.json'
        if f.exists() and json.load(open(fid, encoding='utf-8')) == ids:
            doc_emb[m] = np.load(f)
        else:
            print(f'{m} 문서 {len(docs)}개 임베딩')
            doc_emb[m] = gemini_embed(gclient, m, docs, 'RETRIEVAL_DOCUMENT', log_every=4)
            np.save(f, doc_emb[m]); json.dump(ids, open(fid, 'w', encoding='utf-8'))
        fq = EMB_DIR / f'{m}_queries.json'
        cache = json.load(open(fq, encoding='utf-8')) if fq.exists() else {}
        todo = [q for q in queries if q not in cache]
        if todo:
            for q, v in zip(todo, gemini_embed(gclient, m, todo, 'RETRIEVAL_QUERY')):
                cache[q] = v.tolist()
            json.dump(cache, open(fq, 'w', encoding='utf-8'))
        q_emb[m] = np.asarray([cache[q] for q in queries], dtype='float32')

    rankers = {m: Ranker(doc_emb[m], tfidf, tx) for m in MODELS}
    result = {'models': MODELS, 'topic_questions': [], 'relevance': {}}

    # ── 1) 주제 질문 14개 ──
    print('\n[주제 질문 14개] 정답 강의를 찾았나 (상위 5개)')
    for m in MODELS:
        for mode, label in (('emb', '임베딩만'), ('hybrid', '키워드 결합')):
            by_q = {q: [sources[i] for i in rankers[m].top(q, q_emb[m][j], mode, 5)] for j, q in enumerate(QUESTIONS)}
            r = score_ranking(f'{m} · {label}', by_q, set(sources))
            result['topic_questions'].append(r)
            print(f"  {r['name']:34s} Hit@1 {r['hit1']:2d} | Hit@3 {r['hit3']:2d} / {r['answerable']} | "
                  f"MRR@5 {r['mrr5']:.2f} | 같은 1위 최대 {r['same_top1_max']}회")

    # ── 2) v4 실제 검색 질문 31개: 근거 관련도 ──
    top4 = {m: {k: rankers[m].top(gen[k]['v3_query'], q_emb[m][len(QUESTIONS) + j], 'hybrid', 4)
                for j, k in enumerate(keys)} for m in MODELS}
    same_as_v4 = sum([docs[i] for i in top4['minilm'][k]] == gen[k]['v3_ctx'] for k in keys)
    print(f'\n점검: MiniLM 결과가 실제 v4 생성에 쓴 근거와 같은 쌍 {same_as_v4}/{len(keys)}')

    aclient = anthropic.Anthropic(api_key=dotenv_values(BASE.parent / 'sk-dashboard' / 'backend' / '.env').get('ANTHROPIC_API_KEY', ''))
    usage = {'input': 0, 'output': 0}
    path = OUT / 'embed_relevance_labels.json'
    labels = json.load(open(path, encoding='utf-8')) if path.exists() else {}
    for k in keys:
        union = sorted({i for m in MODELS for i in top4[m][k]})
        if k in labels and all(ids[i] in labels[k] for i in union):
            continue
        random.Random(k).shuffle(union)                        # 순서로 모델이 드러나지 않게
        block = '\n\n'.join(f'[문단{n + 1}] {docs[i][:700]}' for n, i in enumerate(union))
        prompt = (f'[지원자 답변]\n{gen[k]["interview"][:1500]}\n\n아래는 이 답변에 피드백하려고 면접 코칭 자료에서 검색한 '
                  f'문단 {len(union)}개다. 각 문단이 이 답변을 평가·조언하는 근거로 실제로 쓸모 있으면 1, 아니면 0으로 판정하라. '
                  '인사말·잡담·시간 표기 위주이거나 답변 주제와 무관하면 0.\n\n'
                  f'{block}\n\n출력 형식: {{"labels": [0 또는 1 {len(union)}개], "reasons": ["짧은 이유" {len(union)}개]}}')
        for _ in range(4):
            r = aclient.messages.create(model=JUDGE_MODEL, max_tokens=2000, system=SYSTEM,
                                        messages=[{'role': 'user', 'content': prompt}])
            usage['input'] += r.usage.input_tokens; usage['output'] += r.usage.output_tokens
            text = ''.join(b.text for b in r.content if b.type == 'text')
            try:
                out = json.loads(re.search(r'\{.*\}', text, re.S).group(0))
                if len(out['labels']) == len(union):
                    break
            except Exception:
                pass
        else:
            print('  판정 실패:', k); continue
        labels[k] = {ids[i]: int(l) for i, l in zip(union, out['labels'])}
        json.dump(labels, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'  {k:16s} 문단 {len(union)}개 판정')

    print('\n[v4 검색 질문 31개] 키워드 결합 상위 4개 중 근거로 쓸모 있는 문단')
    done = [k for k in keys if k in labels]
    for m in MODELS:
        hit = sum(labels[k][ids[i]] for k in done for i in top4[m][k])
        per = {k: sum(labels[k][ids[i]] for i in top4[m][k]) for k in done}
        result['relevance'][m] = {'relevant': hit, 'total': 4 * len(done), 'rate': round(hit / (4 * len(done)), 3),
                                  'per_pair': per, 'top4_sources': {k: [sources[i] for i in top4[m][k]] for k in done}}
        print(f'  {m:22s} {hit:3d}/{4 * len(done)} ({hit / (4 * len(done)):.0%})')
    for a in MODELS[1:]:
        d = [result['relevance'][a]['per_pair'][k] - result['relevance']['minilm']['per_pair'][k] for k in done]
        result['relevance'][a]['vs_minilm_per_pair'] = {'better': sum(x > 0 for x in d), 'same': sum(x == 0 for x in d), 'worse': sum(x < 0 for x in d)}
        print(f'  {a} vs MiniLM (쌍별): 나음 {sum(x > 0 for x in d)} · 같음 {sum(x == 0 for x in d)} · 못함 {sum(x < 0 for x in d)}')
    result['minilm_same_as_v4_ctx'] = f'{same_as_v4}/{len(keys)}'
    result['judge_usage'] = usage
    result['judge_cost_usd_est'] = round(usage['input'] / 1e6 * PRICE_IN + usage['output'] / 1e6 * PRICE_OUT, 3)
    json.dump(result, open(OUT / 'embed_compare.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('판정 비용 약 $', result['judge_cost_usd_est'])


if __name__ == '__main__':
    main()
