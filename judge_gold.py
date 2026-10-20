"""
judge_gold.py — 판정의 정답 목록 초안 만들기

실제 코치 피드백에서 핵심 지적 2~4개와 전체 평가(잘함/혼합/아쉬움)를 Claude로 뽑아 초안을 만든다.
초안은 사람이 확인·수정해 eval_results_v2/gold.json으로 확정한다 (LLM이 만든 정답으로 LLM을 채점하지 않기 위해).
"""
import json, re

from eval_v2 import BASE, OUT
from judge_v2 import JUDGE_MODEL, PRICE_IN, PRICE_OUT


def main():
    from dotenv import dotenv_values
    import anthropic

    client = anthropic.Anthropic(api_key=dotenv_values(BASE.parent / 'sk-dashboard' / 'backend' / '.env').get('ANTHROPIC_API_KEY', ''))
    gen = json.load(open(OUT / 'generation_outputs.json', encoding='utf-8'))
    usage = {'input': 0, 'output': 0}
    draft = {}
    for k, g in gen.items():
        r = client.messages.create(
            model=JUDGE_MODEL, max_tokens=700, system='너는 면접 코칭 피드백을 정리하는 평가자다. 반드시 JSON 하나만 출력한다.',
            messages=[{'role': 'user', 'content':
                f'[지원자 답변]\n{g["interview"][:1500]}\n\n[실제 코치 피드백]\n{g["coach"][:2500]}\n\n'
                '1) 코치가 이 답변에 대해 짚은 핵심 지적을 2~4개로 정리하라. 각 지적은 "무엇이 좋거나 문제인지 + 어떻게 하라는지"를 '
                '한 문장으로 쓴다. 코치가 말하지 않은 내용은 넣지 않는다. 영상에서만 알 수 있는 태도(외운 티, 긴장 등)는 "[영상]"을 앞에 붙인다.\n'
                '2) 코치의 전체 평가를 잘함 / 혼합 / 아쉬움 중 하나로 고른다.\n'
                '출력 형식: {"key_points": ["...", "..."], "verdict": "잘함|혼합|아쉬움", "verdict_reason": "한 문장"}'}])
        usage['input'] += r.usage.input_tokens; usage['output'] += r.usage.output_tokens
        text = ''.join(b.text for b in r.content if b.type == 'text')
        m = re.search(r'\{.*\}', text, re.S)
        draft[k] = json.loads(m.group(0)) if m else {'raw': text}
        print(f"  {k}: 지적 {len(draft[k].get('key_points', []))}개 · {draft[k].get('verdict')}")
    json.dump(draft, open(OUT / 'gold_draft.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('비용 약 $', round(usage['input'] / 1e6 * PRICE_IN + usage['output'] / 1e6 * PRICE_OUT, 3))


if __name__ == '__main__':
    main()
