"""
judge_keyword_query.py — 실제 서비스처럼 "답변 → 핵심 키워드 5개 → 검색"으로 질의했을 때 v1·v2 검색 관련도 비교

judge_v2.py는 긴 답변 원문을 그대로 질의로 넣었다 (04 텍스트 경로). 05·06 영상 경로는 Gemini가 키워드 5개를
먼저 뽑아 검색하므로, 같은 형태의 질의로 다시 잰다. 키워드 추출·판정 모두 Claude(같은 프롬프트, 판정은 judge_v2와 동일).
"""
import json, re, shutil, tempfile
from pathlib import Path

from eval_v2 import BASE, OUT, PROJ
from judge_v2 import JUDGE_MODEL, PRICE_IN, PRICE_OUT, SYSTEM


def main():
    from dotenv import dotenv_values
    import anthropic
    import chromadb
    from chromadb.utils import embedding_functions
    from rag_v2 import EMBED_MODEL, HybridRetriever

    client = anthropic.Anthropic(api_key=dotenv_values(BASE.parent / 'sk-dashboard' / 'backend' / '.env').get('ANTHROPIC_API_KEY', ''))
    usage = {'input': 0, 'output': 0, 'calls': 0}

    def ask(prompt, system=SYSTEM, max_tokens=600):
        r = client.messages.create(model=JUDGE_MODEL, max_tokens=max_tokens, system=system,
                                   messages=[{'role': 'user', 'content': prompt}])
        usage['input'] += r.usage.input_tokens; usage['output'] += r.usage.output_tokens; usage['calls'] += 1
        return ''.join(b.text for b in r.content if b.type == 'text')

    gen = json.load(open(OUT / 'generation_outputs.json', encoding='utf-8'))
    tmp = Path(tempfile.mkdtemp()) / 'chroma_db_v1'
    shutil.copytree(PROJ / 'chroma_db', tmp)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    v1db = chromadb.PersistentClient(path=str(tmp)).get_collection('interview_rag', embedding_function=ef)
    rag = HybridRetriever()

    res = {}
    for k, g in gen.items():
        answer = g['interview'][:1500]
        kw = ask(f'[면접 답변]\n{answer}\n\n이 답변에서 지원자가 말하는 핵심 키워드를 5개만 추출해줘. 단어만, 쉼표로 구분.',
                 system='짧게 답한다.', max_tokens=60).strip()
        docs = {'v1': v1db.query(query_texts=[kw[:200]], n_results=4)['documents'][0],
                'v2': [h['text'] for h in rag.search(kw, 4)]}
        r = res[k] = {'keywords': kw}
        for ver in ('v1', 'v2'):
            block = '\n\n'.join(f'[문단{i+1}] {d[:700]}' for i, d in enumerate(docs[ver]))
            text = ask(f'[지원자 답변]\n{answer}\n\n아래는 이 답변에 피드백하려고 면접 코칭 강의 자료에서 검색한 문단 4개다. '
                       '각 문단이 이 답변을 평가·조언하는 근거로 실제로 쓸모 있으면 1, 아니면 0으로 판정하라. '
                       '인사말·잡담·시간 표기 위주이거나 답변 주제와 무관하면 0.\n\n'
                       f'{block}\n\n출력 형식: {{"labels": [0 또는 1 네 개], "reasons": ["짧은 이유" 네 개]}}')
            m = re.search(r'\{.*\}', text, re.S)
            r[f'relevance_{ver}'] = json.loads(m.group(0)) if m else {'raw': text}
        print(f"  {k:14s} 키워드: {kw[:40]:42s} | 관련 v1 {sum(r['relevance_v1'].get('labels', []))}/4 · v2 {sum(r['relevance_v2'].get('labels', []))}/4")

    n = len(res)
    summ = {'query': '답변에서 뽑은 키워드 5개', 'items': n,
            'retrieval_precision': {v: f"{sum(sum(r[f'relevance_{v}'].get('labels', [])) for r in res.values())}/{4 * n}" for v in ('v1', 'v2')},
            'usage': usage, 'cost_usd_est': round(usage['input'] / 1e6 * PRICE_IN + usage['output'] / 1e6 * PRICE_OUT, 3)}
    json.dump({'summary': summ, 'items': res}, open(OUT / 'judge_keyword_query.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n', json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
