"""
pipeline_v3.py — v2 + 판정 결과에 따른 보완 두 가지

LLM 판정(judge_v2.py) 결과
- v2 검색은 짧은 주제 질문에서는 강했지만(정답 강의 10/14), 답변 원문(10/24)이나 키워드 5개(12/24)로 검색하면 v1과 차이가 없었다
- 피드백 예시를 무작위로 넣으니 판단이 한쪽('부정')으로 쏠렸다

보완
1) 질의 정제: 답변을 그대로 넣지 않고, LLM이 "이 답변에 필요한 코칭 주제"를 짧은 질문으로 바꿔 검색한다
2) 비슷한 답변 예시: 무작위 5개 대신, 입력 답변과 비슷한 모의면접 답변-피드백 쌍을 골라 예시로 넣는다
"""
import numpy as np

REWRITE_PROMPT = ('다음 면접 답변에 피드백하려면 면접 코칭 강의에서 어떤 주제를 찾아봐야 하는지, '
                  '검색용 질문을 한 문장으로 써줘. 예: "회사 서비스를 써 보지 않은 지원동기는 어떻게 보완하나요?" '
                  '질문 한 문장만 출력해.\n\n[면접 답변]\n')


def rewrite_query(client, model, answer: str) -> str:
    from google.genai import types
    r = client.models.generate_content(model=model, contents=REWRITE_PROMPT + answer[:1500],
                                       config=types.GenerateContentConfig(max_output_tokens=2048))
    return (r.text or '').strip().strip('"') or answer[:300]


class ExampleSelector:
    """입력 답변과 비슷한 답변-피드백 쌍 고르기 (키워드 + 임베딩 순위 결합, rag_v2와 같은 방식)"""

    def __init__(self, pairs: list[dict], k_rrf: int = 60):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sentence_transformers import SentenceTransformer
        from rag_v2 import EMBED_MODEL
        self.pairs = pairs
        texts = [p['interview'] for p in pairs]
        self.tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)).fit(texts)
        self.tx = self.tfidf.transform(texts)
        self.model = SentenceTransformer(EMBED_MODEL)
        self.emb = self.model.encode([t[:500] for t in texts], normalize_embeddings=True)
        self.k_rrf = k_rrf

    def select(self, answer: str, k: int = 5) -> list[dict]:
        s_kw = (self.tx @ self.tfidf.transform([answer]).T).toarray().ravel()
        s_emb = self.emb @ self.model.encode([answer[:500]], normalize_embeddings=True)[0]
        r_kw, r_emb = np.argsort(np.argsort(-s_kw)), np.argsort(np.argsort(-s_emb))
        score = 1 / (self.k_rrf + r_kw) + 1 / (self.k_rrf + r_emb)
        return [self.pairs[i] for i in np.argsort(-score)[:k]]
