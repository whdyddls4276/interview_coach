"""
rag_v2.py — 참고자료 DB(v2) 구축 + 하이브리드 검색

v1(01_rag_build.ipynb)과 다른 점
- 참고자료는 강의 영상 전사만 사용 (data/rag_data, 74개)
  모의면접 피드백 전사는 Few-Shot 예시(data/feedback_data)로만 쓰고 검색 DB에는 넣지 않는다
- 전사 원문의 시간 표기([12.3초 ~ 14.5초], 단어 단위 줄) 제거
- 검색: 키워드(TF-IDF 글자 2~4그램) 순위 + 임베딩 순위를 RRF로 결합
  paraphrase-multilingual-MiniLM은 128토큰까지만 읽고 정확한 용어 매칭에 약해서,
  임베딩만으로는 서로 다른 질문에도 같은 문단이 1위로 나왔다 (재점검 결과는 eval_v2.py)

사용
    from rag_v2 import build_db, HybridRetriever
    build_db()                         # 최초 1회 (chroma_db/ 아래 interview_rag_v2)
    rag = HybridRetriever()
    rag.search('1분 자기소개 준비', k=4)
"""
import re
from pathlib import Path

BASE        = Path(__file__).parent
RAG_DIR     = BASE / 'data' / 'rag_data'
CHROMA_DIR  = BASE / 'chroma_db'
COLLECTION  = 'interview_rag_v2'
EMBED_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
CHUNK_SIZE  = 300
OVERLAP     = 50

SENT_TS = re.compile(r'^\[\d+(?:\.\d+)?초\s*~\s*\d+(?:\.\d+)?초\]\s*', re.M)
WORD_TS = re.compile(r'^\s*\(\d+(?:\.\d+)?\s*~\s*\d+(?:\.\d+)?\)\s.*$', re.M)


def clean(text: str) -> str:
    """Whisper 전사의 시간 표기와 단어 단위 중복 줄 제거"""
    text = WORD_TS.sub('', text)
    text = SENT_TS.sub('', text)
    return re.sub(r'\n{2,}', '\n', text.replace('﻿', '')).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 10]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """문장 경계에 맞춰 chunk_size자 이상이 되면 자르고, 끝 문장 overlap자를 다음 청크에 이어 붙인다 (v1과 같은 방식)"""
    chunks, buf, buf_len = [], [], 0
    for sent in split_sentences(text):
        buf.append(sent)
        buf_len += len(sent)
        if buf_len >= chunk_size:
            chunks.append(' '.join(buf))
            ov, ol = [], 0
            for s in reversed(buf):
                if ol + len(s) > overlap:
                    break
                ov.insert(0, s)
                ol += len(s)
            buf, buf_len = ov, ol
    if buf:
        chunks.append(' '.join(buf))
    return chunks


def load_chunks() -> list[dict]:
    chunks = []
    for f in sorted(RAG_DIR.glob('*/*.txt')):
        text = clean(f.read_text(encoding='utf-8', errors='ignore'))
        for i, c in enumerate(chunk_text(text)):
            chunks.append({'id': f'{f.parent.name}/{f.stem}#{i}', 'text': c,
                           'source': f.stem, 'category': f.parent.name})
    return chunks


def build_db(rebuild: bool = False):
    import chromadb
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild and COLLECTION in [c.name if hasattr(c, 'name') else c for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    # 임베딩 모델을 컬렉션에 같이 기록해 둬야 query_texts로 검색할 때 같은 모델로 질문을 임베딩한다
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    col = client.get_or_create_collection(COLLECTION, metadata={'hnsw:space': 'cosine'}, embedding_function=ef)
    if col.count() and not rebuild:
        print(f'{COLLECTION} 이미 있음: {col.count()}개 청크')
        return col

    chunks = load_chunks()
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode([c['text'] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    for i in range(0, len(chunks), 500):
        part = chunks[i:i + 500]
        col.add(ids=[c['id'] for c in part], documents=[c['text'] for c in part],
                metadatas=[{'source': c['source'], 'category': c['category']} for c in part],
                embeddings=emb[i:i + 500].tolist())
    print(f'{COLLECTION} 구축: 강의 {len({c["source"] for c in chunks})}개 → {col.count()}개 청크')
    return col


class HybridRetriever:
    """키워드 순위 + 임베딩 순위 결합(RRF) 검색"""

    def __init__(self, k_rrf: int = 60):
        import chromadb
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sentence_transformers import SentenceTransformer

        col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
        r = col.get(include=['documents', 'metadatas', 'embeddings'], limit=100000)
        self.np = np
        self.docs, self.metas = r['documents'], r['metadatas']
        self.emb = np.asarray(r['embeddings'], dtype='float32')
        self.mu = self.emb.mean(axis=0)            # 평균 벡터를 빼서 모든 질문에 비슷하게 걸리는 청크를 눌러 준다
        self.tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)).fit(self.docs)
        self.tx = self.tfidf.transform(self.docs)
        self.model = SentenceTransformer(EMBED_MODEL)
        self.k_rrf = k_rrf

    def search(self, query: str, k: int = 4) -> list[dict]:
        np = self.np
        s_kw = (self.tx @ self.tfidf.transform([query]).T).toarray().ravel()
        q = self.model.encode([query], normalize_embeddings=True)[0]
        s_emb = (self.emb - self.mu) @ (q - self.mu)
        rank_kw = np.argsort(np.argsort(-s_kw))
        rank_emb = np.argsort(np.argsort(-s_emb))
        score = 1 / (self.k_rrf + rank_kw) + 1 / (self.k_rrf + rank_emb)
        top = np.argsort(-score)[:k]
        return [{'text': self.docs[i], 'source': self.metas[i]['source'],
                 'category': self.metas[i]['category'], 'score': float(score[i])} for i in top]
