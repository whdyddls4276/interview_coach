"""
judge_v3.py — 판정 기준(rubric)에 따른 LLM-as-judge: v1 / v2 / v3 피드백 정확도

정답 = 실제 코치 피드백의 핵심 지적·전체 평가 (eval_results_v2/gold.json, Claude 초안을 사람이 확인·수정)
판정 모델 = Claude (채점 대상이 Gemini 출력이라 다른 계열). 버전 이름은 숨기고, 같은 판정을 2번 돌려 일치율을 본다.

판정 기준
 ① 핵심 지적 재현율: 정답 지적마다 짚음 1 / 부분 0.5 / 못 짚음 0
    - 짚음: 같은 문제(무엇이 좋거나 부족한지)와 같은 개선 방향을 말함
    - 부분: 같은 문제를 말했지만 이유나 개선 방향이 다름
 ② 평가 방향 일치: AI 피드백의 전체 평가(잘함/혼합/아쉬움)가 정답과 같은지
 ③ 근거 충실도: 답변에 없는 사실(지원자가 하지 않은 말, 없는 경력)을 지어냈는지
 ④ 실행 가능성 1~5: 1 일반론("구체적으로 하세요") / 3 무엇을 고칠지는 있으나 방법이 모호 / 5 이 답변에 맞춘 구체적 수정 방향
 (검색) v3 질의로 찾은 문단 관련도 0/1 — judge_v2·judge_keyword_query와 같은 기준
"""
import json, re, statistics

from eval_v2 import BASE, OUT
from judge_v2 import JUDGE_MODEL, PRICE_IN, PRICE_OUT, SYSTEM

RUBRIC = """판정 기준
1) 핵심 지적: 아래 정답 지적 각각에 대해 AI 피드백이
   - 같은 문제(무엇이 좋거나 부족한지)와 같은 개선 방향을 말했으면 1
   - 같은 문제를 말했지만 이유나 개선 방향이 다르면 0.5
   - 말하지 않았으면 0
   표현이 달라도 뜻이 같으면 같은 것으로 본다. 말투·길이·형식은 고려하지 않는다.
2) 전체 평가: AI 피드백이 답변을 전체적으로 어떻게 평가하는지 잘함 / 혼합 / 아쉬움 중 하나
3) 지어냄: 지원자 답변에 없는 사실(하지 않은 말, 없는 경력·수치)을 사실처럼 말했으면 1, 아니면 0
4) 실행 가능성 1~5: 1 = 일반론("구체적으로 하세요" 수준), 3 = 무엇을 고칠지는 있으나 방법이 모호, 5 = 이 답변 내용에 맞춘 구체적 수정 방향"""


def main():
    from dotenv import dotenv_values
    import anthropic
    from rag_v2 import HybridRetriever

    client = anthropic.Anthropic(api_key=dotenv_values(BASE.parent / 'sk-dashboard' / 'backend' / '.env').get('ANTHROPIC_API_KEY', ''))
    usage = {'input': 0, 'output': 0, 'calls': 0}

    def ask(prompt, max_tokens=700):
        r = client.messages.create(model=JUDGE_MODEL, max_tokens=max_tokens, system=SYSTEM,
                                   messages=[{'role': 'user', 'content': prompt}])
        usage['input'] += r.usage.input_tokens; usage['output'] += r.usage.output_tokens; usage['calls'] += 1
        text = ''.join(b.text for b in r.content if b.type == 'text')
        m = re.search(r'\{.*\}', text, re.S)
        return json.loads(m.group(0)) if m else {'raw': text}

    gold = json.load(open(OUT / 'gold.json', encoding='utf-8'))['items']
    gen = json.load(open(OUT / 'generation_outputs.json', encoding='utf-8'))
    versions = [v for v in ('v1', 'v2', 'v3') if all(gen[k].get(v) for k in gold)]
    rag = HybridRetriever()

    path = OUT / 'judge_v3.json'
    res = json.load(open(path, encoding='utf-8')) if path.exists() else {}   # 이미 채점한 (건, 버전)은 다시 묻지 않는다
    for k, gd in gold.items():
        g = gen[k]
        answer = g['interview'][:1500]
        pts = '\n'.join(f'{i+1}. {p}' for i, p in enumerate(gd['key_points']))
        r = res.setdefault(k, {})
        for ver in versions:
            if len(r.get(ver, [])) == 2:
                continue
            runs = []
            for rep in range(2):
                runs.append(ask(
                    f'[지원자 답변]\n{answer}\n\n[정답 — 실제 코치가 짚은 핵심 지적]\n{pts}\n\n[AI 피드백]\n{g[ver][:2500]}\n\n{RUBRIC}\n\n'
                    f'출력 형식: {{"coverage": [정답 지적 {len(gd["key_points"])}개 각각 1 또는 0.5 또는 0], '
                    '"verdict": "잘함|혼합|아쉬움", "unsupported": 0 또는 1, "actionability": 1~5, "reason": "두 문장 이내"}'))
            r[ver] = runs
        if 'v3' in versions and 'relevance_v3' not in r:   # v3 질의로 찾은 문단 관련도
            docs = '\n\n'.join(f'[문단{i+1}] {h["text"][:700]}' for i, h in enumerate(rag.search(g['v3_query'], 4)))
            r['relevance_v3'] = ask(
                f'[지원자 답변]\n{answer}\n\n아래는 이 답변에 피드백하려고 면접 코칭 강의 자료에서 검색한 문단 4개다. '
                '각 문단이 이 답변을 평가·조언하는 근거로 실제로 쓸모 있으면 1, 아니면 0으로 판정하라. '
                '인사말·잡담·시간 표기 위주이거나 답변 주제와 무관하면 0.\n\n'
                f'{docs}\n\n출력 형식: {{"labels": [0 또는 1 네 개], "reasons": ["짧은 이유" 네 개]}}')
        line = ' | '.join(f"{v} 재현 {sum(r[v][0].get('coverage', []))}/{len(gd['key_points'])} {r[v][0].get('verdict')}" for v in versions)
        print(f'  {k:14s} 정답 {gd["verdict"]} | {line}')
        json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    summ = {'judge_model': JUDGE_MODEL, 'items': len(gold), 'key_points': sum(len(v['key_points']) for v in gold.values()),
            'versions': {}}
    for ver in versions:
        rec, ver_ok, uns, act, agree_cov, agree_ver = [], 0, 0, [], 0, 0
        for k, gd in gold.items():
            a, b = res[k][ver]
            ca, cb = a.get('coverage', []), b.get('coverage', [])
            rec.append((sum(ca) + sum(cb)) / 2 / len(gd['key_points']))
            ver_ok += (a.get('verdict') == gd['verdict']) + (b.get('verdict') == gd['verdict'])
            uns += a.get('unsupported', 0) + b.get('unsupported', 0)
            act += [a.get('actionability', 0), b.get('actionability', 0)]
            agree_cov += ca == cb; agree_ver += a.get('verdict') == b.get('verdict')
        n = len(gold)
        summ['versions'][ver] = {'key_point_recall': round(statistics.mean(rec), 3),
                                 'verdict_match': f'{ver_ok / 2:.1f}/{n}', 'unsupported': f'{uns / 2:.1f}/{n}',
                                 'actionability_avg': round(statistics.mean(act), 2),
                                 'judge_self_agreement': {'coverage_same': f'{agree_cov}/{n}', 'verdict_same': f'{agree_ver}/{n}'}}
    if 'v3' in versions:
        summ['retrieval_precision_v3_query'] = f"{sum(sum(res[k]['relevance_v3'].get('labels', [])) for k in gold)}/{4 * len(gold)}"
    summ['usage'] = usage
    summ['cost_usd_est'] = round(usage['input'] / 1e6 * PRICE_IN + usage['output'] / 1e6 * PRICE_OUT, 3)
    json.dump(summ, open(OUT / 'judge_v3_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n[판정 기준 기반 LLM 판정 요약]', json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
