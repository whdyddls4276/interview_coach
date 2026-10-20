"""
compare_human.py — 사람 판정(human_check.md)과 Claude 판정 비교

1) 코치 지적 재현: 본인 짚음/부분/놓침 vs judge_full.py의 v4 coverage(2회) — 일치율, 가중 카파
2) 전체 평가: 본인 잘함/혼합/아쉬움 vs Claude가 본 AI 피드백의 평가(2회)
3) 지어냄: judge_unsupported.py가 '사실처럼'으로 잡은 v4 문장 중 본인이 '맞음'이라 한 비율 (판정 정밀도)
채우지 않은 칸(___)은 건너뛴다.
"""
import json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
FULL = Path(__file__).parent / 'eval_results_v2' / 'full'
COV = {'짚음': 1.0, '부분': 0.5, '놓침': 0.0}
VERDICTS = ('잘함', '혼합', '아쉬움')


def parse(md: str):
    cov, ver, fab = [], [], []
    key, point, part3 = None, None, False
    for line in md.splitlines():
        if line.startswith('# 3부'):
            part3 = True
        m = re.match(r'^##+ (?:\d+|3-\d+)\. `([^`]+)`', line)
        if m:
            key, point = m.group(1), None
            continue
        m = re.match(r'^- 지적 (\d+)\.', line)
        if m:
            point = int(m.group(1)) - 1
            continue
        m = re.match(r'^- AI 피드백 문장: "(.*)"$', line)
        if m:
            quote = m.group(1)
            continue
        m = re.match(r'^\s*- 본인:\s*(\S+)', line)
        if not m or m.group(1).startswith('___'):
            continue
        v = m.group(1)
        if part3:
            fab.append((key, quote, v))
        elif point is not None:
            cov.append((key, point, v)); point = None
        else:
            ver.append((key, v))
    return cov, ver, fab


def kappa(pairs, cats, weighted=False):
    """두 평가자(사람, Claude)의 카파. weighted=True면 짚음-부분-놓침 순서를 반영한 선형 가중"""
    n = len(pairs)
    if not n:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    w = lambda a, b: 1 - abs(idx[a] - idx[b]) / (k - 1) if weighted else float(a == b)
    po = sum(w(a, b) for a, b in pairs) / n
    pa = {c: sum(a == c for a, _ in pairs) / n for c in cats}
    pb = {c: sum(b == c for _, b in pairs) / n for c in cats}
    pe = sum(pa[a] * pb[b] * w(a, b) for a in cats for b in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def main():
    md = (FULL / 'human_check.md').read_text(encoding='utf-8')
    cov, ver, fab = parse(md)
    judge = json.load(open(FULL / 'judge.json', encoding='utf-8'))
    name = {1.0: '짚음', 0.5: '부분', 0.0: '놓침'}
    out = {}

    rows = []
    for k, p, v in cov:
        if v not in COV:
            print('  알 수 없는 값 건너뜀:', k, p + 1, v); continue
        runs = [r['coverage'][p] for r in judge[k]['v4'] if p < len(r.get('coverage', []))]
        rows.append((k, p, COV[v], runs))
    if rows:
        exact = [sum(h == c for c in runs) / len(runs) for k, p, h, runs in rows]
        pairs = [(name[h], name[c]) for k, p, h, runs in rows for c in runs]
        out['coverage'] = {
            'n_points': len(rows),
            'exact_agree': round(sum(exact) / len(exact), 3),
            'within_half': round(sum(abs(h - c) <= 0.5 for k, p, h, runs in rows for c in runs) / len(pairs), 3),
            'kappa_weighted': kappa(pairs, ['놓침', '부분', '짚음'], weighted=True),
            'human_recall': round(sum(h for _, _, h, _ in rows) / len(rows), 3),
            'claude_recall': round(sum(sum(r) / len(r) for *_, r in rows) / len(rows), 3),
            'disagreements': [{'pair': k, 'point': p + 1, 'human': name[h], 'claude': [name[c] for c in runs]}
                              for k, p, h, runs in rows if any(h != c for c in runs)],
        }
    vrows = []
    for k, v in ver:
        if v in VERDICTS:
            vrows.append((k, v, [r.get('verdict') for r in judge[k]['v4']]))
    if vrows:
        out['verdict'] = {'n': len(vrows),
                          'exact_agree': round(sum(sum(v == c for c in cs) / len(cs) for k, v, cs in vrows) / len(vrows), 3),
                          'kappa': kappa([(v, c) for k, v, cs in vrows for c in cs], list(VERDICTS)),
                          'rows': [{'pair': k, 'human': v, 'claude': cs} for k, v, cs in vrows]}
    if fab:
        cnt = {x: sum(v == x for *_, v in fab) for x in ('맞음', '아님', '애매')}
        out['fabrication_flags'] = {'n': len(fab), **cnt,
                                    'precision': round(cnt['맞음'] / max(1, cnt['맞음'] + cnt['아님']), 3),
                                    'rows': [{'pair': k, 'quote': q[:80], 'human': v} for k, q, v in fab]}
    json.dump(out, open(FULL / 'human_check_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('disagreements', 'rows')} for k, v in out.items()},
                     ensure_ascii=False, indent=1))
    if not out:
        print('아직 채운 칸이 없습니다.')


if __name__ == '__main__':
    main()
