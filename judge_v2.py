"""
judge_v2.py — LLM-as-judge로 v1 vs v2 생성·검색 품질 비교 (eval_v2.py 생성 결과 재사용)

채점 대상(v1·v2 피드백)은 Gemini 출력이라, 같은 계열 편향을 피하려고 판정은 Claude로 한다.
1) 검색 관련도: 실제 지원자 답변으로 검색한 근거 4개가 피드백 근거로 쓸모 있는지 0/1 → 정밀도
   (eval_v2의 검색 평가는 강의 제목에서 만든 질문이라 키워드 검색에 유리 → 실제 답변 질의로 보완)
2) 코치와 가까운 쪽: 실제 코치 피드백을 기준으로 v1·v2 중 내용·판단이 가까운 쪽 선택
   A/B 순서를 바꿔 두 번 묻고, 두 번 모두 같은 쪽을 고른 경우만 승리로 센다 (위치 편향 확인)
3) 근거 활용: 검색 근거의 관점을 답변에 맞게 썼는지 1~5, 답변·근거에 없는 사실을 지어냈는지 0/1

실행: python judge_v2.py   (eval_v2.py를 먼저 실행해 eval_results_v2/generation_outputs.json이 있어야 함)
API 키: sk-dashboard/backend/.env 의 ANTHROPIC_API_KEY
"""
import json, os, random, re, shutil, sys, tempfile
from pathlib import Path

from eval_v2 import BASE, OUT, PROJ, load_pairs

JUDGE_MODEL = 'claude-sonnet-4-6'
PRICE_IN, PRICE_OUT = 3.0, 15.0          # USD / 1M 토큰 (Sonnet 계열 표준 단가 기준 추정)
SYSTEM = '너는 면접 코칭 AI의 결과를 채점하는 평가자다. 반드시 JSON 하나만 출력한다.'


def main():
    from dotenv import dotenv_values
    import anthropic
    import chromadb
    from chromadb.utils import embedding_functions
    from rag_v2 import EMBED_MODEL, HybridRetriever

    client = anthropic.Anthropic(api_key=dotenv_values(BASE.parent / 'sk-dashboard' / 'backend' / '.env').get('ANTHROPIC_API_KEY', ''))
    usage = {'input': 0, 'output': 0, 'calls': 0}

    def ask(prompt, max_tokens=600):
        r = client.messages.create(model=JUDGE_MODEL, max_tokens=max_tokens, system=SYSTEM,
                                   messages=[{'role': 'user', 'content': prompt}])
        usage['input'] += r.usage.input_tokens; usage['output'] += r.usage.output_tokens; usage['calls'] += 1
        text = ''.join(b.text for b in r.content if b.type == 'text')
        m = re.search(r'\{.*\}', text, re.S)
        return json.loads(m.group(0)) if m else {'raw': text}

    gen = json.load(open(OUT / 'generation_outputs.json', encoding='utf-8'))
    items = {k: v for k, v in gen.items() if v.get('v1') and v.get('v2')}

    # 생성 때와 같은 방식으로 검색 근거를 다시 뽑는다 (v1: 답변 앞 200자 임베딩 / v2: 답변 앞 300자 하이브리드)
    tmp = Path(tempfile.mkdtemp()) / 'chroma_db_v1'
    shutil.copytree(PROJ / 'chroma_db', tmp)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    v1db = chromadb.PersistentClient(path=str(tmp)).get_collection('interview_rag', embedding_function=ef)
    rag = HybridRetriever()
    ctx = {k: {'v1': v1db.query(query_texts=[g['interview'][:200]], n_results=4)['documents'][0],
               'v2': [h['text'] for h in rag.search(g['interview'][:300], 4)]} for k, g in items.items()}

    res = {}
    rng = random.Random(0)
    for k, g in items.items():
        r = res.setdefault(k, {})
        answer = g['interview'][:1500]
        for ver in ('v1', 'v2'):
            docs = '\n\n'.join(f'[문단{i+1}] {d[:700]}' for i, d in enumerate(ctx[k][ver]))
            r[f'relevance_{ver}'] = ask(
                f'[지원자 답변]\n{answer}\n\n아래는 이 답변에 피드백하려고 면접 코칭 강의 자료에서 검색한 문단 4개다. '
                '각 문단이 이 답변을 평가·조언하는 근거로 실제로 쓸모 있으면 1, 아니면 0으로 판정하라. '
                '인사말·잡담·시간 표기 위주이거나 답변 주제와 무관하면 0.\n\n'
                f'{docs}\n\n출력 형식: {{"labels": [0 또는 1 네 개], "reasons": ["짧은 이유" 네 개]}}')
            r[f'grounding_{ver}'] = ask(
                f'[지원자 답변]\n{answer}\n\n[검색한 강의 문단]\n{docs}\n\n[AI 피드백]\n{g[ver][:2500]}\n\n'
                '1) 피드백이 강의 문단의 관점·개념을 이 답변에 맞게 활용했는지 1~5점 (문단이 쓸모없어 활용할 수 없었다면 1). '
                '2) 답변과 강의 문단 어디에도 없는 사실을 지어냈으면 1, 아니면 0.\n'
                '출력 형식: {"uses_reference": 점수, "unsupported": 0 또는 1, "reason": "한두 문장"}')
        # 순서를 바꿔 두 번 묻는다
        first_v2 = rng.random() < 0.5
        for order in ('o1', 'o2'):
            a_is_v2 = first_v2 if order == 'o1' else not first_v2
            A, B = (g['v2'], g['v1']) if a_is_v2 else (g['v1'], g['v2'])
            out = ask(
                f'[지원자 답변]\n{answer}\n\n[실제 코치 피드백 — 기준]\n{g["coach"][:2000]}\n\n'
                f'[피드백 A]\n{A[:2500]}\n\n[피드백 B]\n{B[:2500]}\n\n'
                '기준 피드백과 비교해, 무엇을 잘했고 무엇이 문제이며 어떻게 고치라고 했는지(내용과 판단)가 더 가까운 쪽을 고르라. '
                '말투·길이·형식은 고려하지 마라.\n출력 형식: {"winner": "A" 또는 "B" 또는 "tie", "reason": "한두 문장"}')
            w = out.get('winner')
            out['winner_version'] = 'tie' if w == 'tie' else ('v2' if (w == 'A') == a_is_v2 else 'v1')
            r[f'pairwise_{order}'] = out
        print(f"  {k:14s} 관련도 v1 {sum(r['relevance_v1'].get('labels', []))}/4 · v2 {sum(r['relevance_v2'].get('labels', []))}/4 | "
              f"근거활용 v1 {r['grounding_v1'].get('uses_reference')} · v2 {r['grounding_v2'].get('uses_reference')} | "
              f"코치에 가까운 쪽 {r['pairwise_o1']['winner_version']} / {r['pairwise_o2']['winner_version']}")
        json.dump(res, open(OUT / 'judge.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    n = len(res)
    rel = {v: sum(sum(r[f'relevance_{v}'].get('labels', [])) for r in res.values()) for v in ('v1', 'v2')}
    gnd = {v: round(sum(r[f'grounding_{v}'].get('uses_reference', 0) for r in res.values()) / n, 2) for v in ('v1', 'v2')}
    uns = {v: sum(r[f'grounding_{v}'].get('unsupported', 0) for r in res.values()) for v in ('v1', 'v2')}
    pair = [(r['pairwise_o1']['winner_version'], r['pairwise_o2']['winner_version']) for r in res.values()]
    consistent = [a for a, b in pair if a == b]
    summ = {'judge_model': JUDGE_MODEL, 'items': n,
            'retrieval_precision': {v: f'{rel[v]}/{4 * n}' for v in rel},
            'uses_reference_avg_1to5': gnd, 'unsupported_count': {v: f'{uns[v]}/{n}' for v in uns},
            'pairwise_closer_to_coach': {'v2_wins': consistent.count('v2'), 'v1_wins': consistent.count('v1'),
                                         'tie': consistent.count('tie'), 'inconsistent_order': n - len(consistent)},
            'usage': usage, 'cost_usd_est': round(usage['input'] / 1e6 * PRICE_IN + usage['output'] / 1e6 * PRICE_OUT, 3)}
    json.dump(summ, open(OUT / 'judge_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n[LLM 판정 요약]', json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
