"""
rag_v3.py — 참고자료 DB(v3): 임베딩을 MiniLM → Gemini 임베딩 API(gemini-embedding-001)로 교체

v2(rag_v2.py)와 같은 강의 청크 567개, 같은 검색 방식(키워드 순위 + 임베딩 순위 RRF 결합)이고 임베딩만 다르다.
- MiniLM은 128토큰까지만 읽어 청크의 89%가 잘린 채 검색됐다 (청크 길이 중앙값 165토큰)
- gemini-embedding-001은 2,048토큰까지 읽는다. 문서는 RETRIEVAL_DOCUMENT, 질문은 RETRIEVAL_QUERY로 임베딩
- 비교(embed_compare.py): 주제 질문 14개 Hit@3 10 → 14, 실제 검색 질문 31개 근거 관련도 40% → 60%
  피드백까지 만든 비교(eval_v5.py): 근거 관련도 47% → 55%, 코치 지적 재현율 51% → 52%
v2 컬렉션(interview_rag_v2)은 그대로 두고 같은 chroma_db 폴더에 interview_rag_v3를 따로 만든다.

사용
    from rag_v3 import build_db, HybridRetrieverV3
    build_db(client)                   # 최초 1회 (Gemini 임베딩 API 호출, 약 $0.02)
    rag = HybridRetrieverV3(client)
    rag.search('1분 자기소개 준비', k=4)
"""
from rag_v2 import CHROMA_DIR, load_chunks
from embed_gemini import gemini_embed

COLLECTION = 'interview_rag_v3'
EMBED_MODEL = 'gemini-embedding-001'


def build_db(client, rebuild: bool = False):
    import chromadb

    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild and COLLECTION in [c.name if hasattr(c, 'name') else c for c in db.list_collections()]:
        db.delete_collection(COLLECTION)
    # 임베딩은 직접 계산해 넣고 검색도 벡터로 하므로 컬렉션에 임베딩 함수를 두지 않는다 (모델 이름은 메타데이터에 기록)
    col = db.get_or_create_collection(COLLECTION, metadata={'hnsw:space': 'cosine', 'embed_model': EMBED_MODEL})
    if col.count() and not rebuild:
        print(f'{COLLECTION} 이미 있음: {col.count()}개 청크')
        return col

    chunks = load_chunks()
    emb = gemini_embed(client, EMBED_MODEL, [c['text'] for c in chunks], 'RETRIEVAL_DOCUMENT')
    for i in range(0, len(chunks), 500):
        part = chunks[i:i + 500]
        col.add(ids=[c['id'] for c in part], documents=[c['text'] for c in part],
                metadatas=[{'source': c['source'], 'category': c['category']} for c in part],
                embeddings=emb[i:i + 500].tolist())
    print(f'{COLLECTION} 구축: 강의 {len({c["source"] for c in chunks})}개 → {col.count()}개 청크 ({EMBED_MODEL})')
    return col


class HybridRetrieverV3:
    """키워드 순위 + Gemini 임베딩 순위 결합(RRF) 검색 — rag_v2.HybridRetriever와 같은 계산"""

    def __init__(self, client, k_rrf: int = 60):
        import chromadb
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
        r = col.get(include=['documents', 'metadatas', 'embeddings'], limit=100000)
        self.np, self.client = np, client
        self.docs, self.metas = r['documents'], r['metadatas']
        self.emb = np.asarray(r['embeddings'], dtype='float32')
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True)
        self.mu = self.emb.mean(axis=0)            # 평균 벡터를 빼서 모든 질문에 비슷하게 걸리는 청크를 눌러 준다
        self.tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)).fit(self.docs)
        self.tx = self.tfidf.transform(self.docs)
        self.k_rrf = k_rrf

    def search(self, query: str, k: int = 4) -> list[dict]:
        np = self.np
        s_kw = (self.tx @ self.tfidf.transform([query]).T).toarray().ravel()
        q = gemini_embed(self.client, EMBED_MODEL, [query], 'RETRIEVAL_QUERY')[0]
        s_emb = (self.emb - self.mu) @ (q - self.mu)
        rank_kw = np.argsort(np.argsort(-s_kw))
        rank_emb = np.argsort(np.argsort(-s_emb))
        score = 1 / (self.k_rrf + rank_kw) + 1 / (self.k_rrf + rank_emb)
        top = np.argsort(-score)[:k]
        return [{'text': self.docs[i], 'source': self.metas[i]['source'],
                 'category': self.metas[i]['category'], 'score': float(score[i])} for i in top]
