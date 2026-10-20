"""
embed_gemini.py — Gemini 임베딩 API 호출 (문서/질문 구분, 묶음 요청, 재시도)

gemini-embedding-2는 문자열 목록을 그대로 보내면 하나로 합쳐 벡터 1개를 돌려준다
→ 문장마다 types.Content로 감싸서 보낸다 (하나씩 보낸 결과와 같음을 확인).
"""
import time
import numpy as np


def gemini_embed(client, model: str, texts: list[str], task: str = 'RETRIEVAL_DOCUMENT',
                 batch: int = 50, log_every: int = 0) -> np.ndarray:
    from google.genai import types
    cfg = types.EmbedContentConfig(task_type=task)
    out = []
    for i in range(0, len(texts), batch):
        part = [types.Content(parts=[types.Part(text=t)]) for t in texts[i:i + batch]]
        for k in range(6):
            try:
                r = client.models.embed_content(model=model, contents=part, config=cfg)
                break
            except Exception as e:
                wait = 5 * 2 ** k
                print(f'   임베딩 재시도 {k + 1} ({wait}s): {str(e)[:80]}')
                time.sleep(wait)
        else:
            raise RuntimeError('임베딩 요청이 계속 실패')
        assert len(r.embeddings) == len(part), '요청한 문장 수와 벡터 수가 다름'
        out += [e.values for e in r.embeddings]
        if log_every and (i // batch) % log_every == 0:
            print(f'   {min(i + batch, len(texts))}/{len(texts)}')
    v = np.asarray(out, dtype='float32')
    return v / np.linalg.norm(v, axis=1, keepdims=True)
