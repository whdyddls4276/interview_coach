"""
judge_full.py — 31쌍 전체 LLM-as-judge (판정 기준은 judge_v3.py와 동일)

정답: eval_results_v2/full/gold.json (Claude 초안 → 사람 확인·수정)
대상: eval_results_v2/full/generation.json 의 v1 / v2 / v3
판정: Claude, 버전 이름 숨김, (쌍, 버전)마다 2번 → 평균 · 두 번의 일치율
검색: 각 버전이 실제로 넣은 근거 4개(v1_ctx / v2_ctx / v3_ctx)의 관련도 0/1
결과: full/judge.json, full/judge_summary.json
"""
import json, re, statistics, threading
from concurrent.futures import ThreadPoolExecutor

from eval_v2 import BASE, OUT
from judge_v2 import JUDGE_MODEL, PRICE_IN, PRICE_OUT, SYSTEM
from judge_v3 import RUBRIC

FULL = OUT / 'full'
VERSIONS = ('v1', 'v2', 'v3', 'v4', 'v5')   # v4 = v3 입력 + v1 프롬프트 (eval_v4.py) / v5 = v4 + Gemini 임베딩 검색 (eval_v5.py)
lock = threading.Lock()


def main():
    from dotenv import dotenv_values
    import anthropic

    client = anthropic.Anthropic(api_key=dotenv_values(BASE.parent / 'sk-dashboard' / 'backend' / '.env').get('ANTHROPIC_API_KEY', ''))
    usage = {'input': 0, 'output': 0, 'calls': 0}

    def ask(prompt, max_tokens=700):
        for k in range(4):
            try:
                r = client.messages.create(model=JUDGE_MODEL, max_tokens=max_tokens, system=SYSTEM,
                                           messages=[{'role': 'user', 'content': prompt}])
                break
            except Exception as e:
                print('   재시도:', str(e)[:60])
        else:
            return {}
        with lock:
            usage['input'] += r.usage.input_tokens; usage['output'] += r.usage.output_tokens; usage['calls'] += 1
        text = ''.join(b.text for b in r.content if b.type == 'text')
        m = re.search(r'\{.*\}', text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}

    gold = json.load(open(FULL / 'gold.json', encoding='utf-8'))['items']
    gen = json.load(open(FULL / 'generation.json', encoding='utf-8'))
    path = FULL / 'judge.json'
    res = json.load(open(path, encoding='utf-8')) if path.exists() else {}

    def judge_item(k):
        gd, g = gold[k], gen[k]
        answer = g['interview'][:1500]
        pts = '\n'.join(f'{i+1}. {p}' for i, p in enumerate(gd['key_points']))
        with lock:
            r = res.setdefault(k, {})
        for ver in VERSIONS:
            if len(r.get(ver, [])) < 2:
                r[ver] = [ask(
                    f'[지원자 답변]\n{answer}\n\n[정답 — 실제 코치가 짚은 핵심 지적]\n{pts}\n\n[AI 피드백]\n{g[ver][:2500]}\n\n{RUBRIC}\n\n'
                    f'출력 형식: {{"coverage": [정답 지적 {len(gd["key_points"])}개 각각 1 또는 0.5 또는 0], '
                    '"verdict": "잘함|혼합|아쉬움", "unsupported": 0 또는 1, "actionability": 1~5, "reason": "두 문장 이내"}') for _ in range(2)]
            if f'relevance_{ver}' not in r and ver == 'v4' and 'relevance_v3' in r and g.get('v4_ctx') == g.get('v3_ctx'):
                r['relevance_v4'] = r['relevance_v3']   # v4는 v3와 같은 근거를 쓰므로 판정 재사용
            if f'relevance_{ver}' not in r:
                docs ='\n\n'.join(f'[문단{i+1}] {d[:700]}' for i, d in enumerate(g[f'{ver}_ctx']))
                r[f'relevance_{ver}'] = ask(
                    f'[지원자 답변]\n{answer}\n\n아래는 이 답변에 피드백하려고 면접 코칭 자료에서 검색한 문단 4개다. '
                    '각 문단이 이 답변을 평가·조언하는 근거로 실제로 쓸모 있으면 1, 아니면 0으로 판정하라. '
                    '인사말·잡담·시간 표기 위주이거나 답변 주제와 무관하면 0.\n\n'
                    f'{docs}\n\n출력 형식: {{"labels": [0 또는 1 네 개], "reasons": ["짧은 이유" 네 개]}}')
        with lock:
            json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f"  {k:16s} " + ' | '.join(f"{v} {sum(r[v][0].get('coverage', []))}/{len(gd['key_points'])}" for v in VERSIONS))

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(judge_item, [k for k in gold if all(gen.get(k, {}).get(v) for v in VERSIONS)]))

    # ── 요약 ──
    keys = [k for k in gold if all(len(res.get(k, {}).get(v, [])) == 2 for v in VERSIONS)]
    def recall(k, ver, video=True):
        pts = gold[k]['key_points']
        idx = [i for i, p in enumerate(pts) if video or not p.startswith('[영상]')]
        if not idx:
            return None
        vals = []
        for run in res[k][ver]:
            cov = run.get('coverage', [])
            vals.append(sum(cov[i] for i in idx if i < len(cov)) / len(idx))
        return statistics.mean(vals)
    summ = {'judge_model': JUDGE_MODEL, 'pairs': len(keys), 'key_points': sum(len(gold[k]['key_points']) for k in keys), 'versions': {}}
    per_item = {}
    for ver in VERSIONS:
        rec = [recall(k, ver) for k in keys]
        rec_txt = [x for x in (recall(k, ver, video=False) for k in keys) if x is not None]
        runs = [(k, run) for k in keys for run in res[k][ver]]
        vm = sum(run.get('verdict') == gold[k]['verdict'] for k, run in runs) / 2
        uns = sum(run.get('unsupported', 0) for k, run in runs) / 2
        act = statistics.mean(run.get('actionability', 0) for k, run in runs)
        same_cov = sum(res[k][ver][0].get('coverage') == res[k][ver][1].get('coverage') for k in keys)
        same_ver = sum(res[k][ver][0].get('verdict') == res[k][ver][1].get('verdict') for k in keys)
        rel = sum(sum(res[k][f'relevance_{ver}'].get('labels', [])) for k in keys)
        summ['versions'][ver] = {'key_point_recall': round(statistics.mean(rec), 3),
                                 'key_point_recall_text_only': round(statistics.mean(rec_txt), 3),
                                 'verdict_match': f'{vm:.1f}/{len(keys)}', 'unsupported': f'{uns:.1f}/{len(keys)}',
                                 'actionability_avg': round(act, 2), 'retrieval_precision': f'{rel}/{4 * len(keys)}',
                                 'judge_two_runs_same': {'coverage': f'{same_cov}/{len(keys)}', 'verdict': f'{same_ver}/{len(keys)}'}}
        for k, x in zip(keys, rec):
            per_item.setdefault(k, {})[ver] = round(x, 3)
    for a, b in (('v2', 'v1'), ('v3', 'v1'), ('v3', 'v2'), ('v4', 'v1'), ('v4', 'v3'), ('v5', 'v4'), ('v5', 'v1')):
        d = [per_item[k][a] - per_item[k][b] for k in keys]
        summ[f'{a}_vs_{b}_recall_per_pair'] = {'better': sum(x > 1e-9 for x in d), 'same': sum(abs(x) <= 1e-9 for x in d), 'worse': sum(x < -1e-9 for x in d)}
    summ['per_pair_recall'] = per_item
    summ['usage'] = usage
    summ['cost_usd_est'] = round(usage['input'] / 1e6 * PRICE_IN + usage['output'] / 1e6 * PRICE_OUT, 3)
    json.dump(summ, open(FULL / 'judge_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n[31쌍 LLM 판정 요약]', json.dumps({k: v for k, v in summ.items() if k != 'per_pair_recall'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    main()
