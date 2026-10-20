"""
judge_gold_full.py — 31쌍 전체의 정답 초안 (이미 사람이 확인한 gold.json 6건은 그대로 두고 나머지만 Claude로 초안)
결과: eval_results_v2/full/gold_draft.json → 사람 확인 후 eval_results_v2/full/gold.json
"""
import json, re

from eval_v2 import BASE, OUT, load_pairs
from judge_v2 import JUDGE_MODEL, PRICE_IN, PRICE_OUT


def main():
    from dotenv import dotenv_values
    import anthropic

    client = anthropic.Anthropic(api_key=dotenv_values(BASE.parent / 'sk-dashboard' / 'backend' / '.env').get('ANTHROPIC_API_KEY', ''))
    done = json.load(open(OUT / 'gold.json', encoding='utf-8'))['items']
    (OUT / 'full').mkdir(exist_ok=True)
    usage = {'input': 0, 'output': 0}
    draft = {}
    for p in load_pairs():
        if p['id'] in done:
            continue
        r = client.messages.create(
            model=JUDGE_MODEL, max_tokens=700, system='너는 면접 코칭 피드백을 정리하는 평가자다. 반드시 JSON 하나만 출력한다.',
            messages=[{'role': 'user', 'content':
                f'[지원자 답변]\n{p["interview"][:1500]}\n\n[실제 코치 피드백]\n{p["feedback"][:2500]}\n\n'
                '1) 코치가 이 답변에 대해 짚은 핵심 지적을 2~4개로 정리하라. 각 지적은 "무엇이 좋거나 문제인지 + 어떻게 하라는지"를 '
                '한 문장으로 쓴다. 코치가 말하지 않은 내용은 넣지 않는다(답변만 보고 추론한 지적 금지). '
                '영상에서만 알 수 있는 태도(외운 티, 긴장 등)는 "[영상]"을 앞에 붙인다.\n'
                '2) 코치의 전체 평가를 잘함 / 혼합 / 아쉬움 중 하나로 고른다. 강점을 분명히 인정하면서 보완점을 말하면 혼합.\n'
                '출력 형식: {"key_points": ["...", "..."], "verdict": "잘함|혼합|아쉬움", "verdict_reason": "한 문장"}'}])
        usage['input'] += r.usage.input_tokens; usage['output'] += r.usage.output_tokens
        text = ''.join(b.text for b in r.content if b.type == 'text')
        m = re.search(r'\{.*\}', text, re.S)
        draft[p['id']] = json.loads(m.group(0)) if m else {'raw': text}
        print(f"  {p['id']}: 지적 {len(draft[p['id']].get('key_points', []))}개 · {draft[p['id']].get('verdict')}")
    json.dump(draft, open(OUT / 'full' / 'gold_draft.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('비용 약 $', round(usage['input'] / 1e6 * PRICE_IN + usage['output'] / 1e6 * PRICE_OUT, 3))


if __name__ == '__main__':
    main()
