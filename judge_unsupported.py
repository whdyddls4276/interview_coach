"""
judge_unsupported.py — 지어냄 판정만 따로 다시 (judge_full.py의 ③을 대체)

judge_full.py는 정답(코치 지적)과 AI 피드백을 한 프롬프트에 함께 넣었다. 그 결과 지어냄 판정 19건 중 10건이
코치 피드백에만 있는 내용("복싱동아리 회장", "유입률 32%")을 AI 피드백이 말한 것으로 착각한 오류였다
(생성된 피드백 어디에도 없음). 또 답변을 1,500자에서 잘라 넣어 뒷부분 내용이 '없는 말'로 보일 수 있었다.

바꾼 점
- 판정 LLM에 정답·코치 피드백을 보여주지 않는다: 지원자 답변 전문 + AI 피드백만
- 지어낸 문장을 AI 피드백에서 그대로 옮겨 적게 하고, 그 문장이 실제 피드백에 있는지 코드로 확인 (없으면 버림)
- '사실처럼'(지원자가 하지 않은 말·없는 경력·수치를 지원자의 것으로 말함)과
  '가상 예시'(고쳐 말하기 예시에 넣은 가상의 수치·사례)를 나눠 센다

대상: eval_results_v2/full/generation.json 의 v1~v4, (쌍, 버전)마다 2번
결과: full/judge_unsupported.json, full/judge_unsupported_summary.json
"""
import json, re, statistics, threading
from concurrent.futures import ThreadPoolExecutor

from eval_v2 import BASE, OUT
from judge_v2 import JUDGE_MODEL, PRICE_IN, PRICE_OUT, SYSTEM

FULL = OUT / 'full'
VERSIONS = ('v1', 'v2', 'v3', 'v4', 'v5')
lock = threading.Lock()

RULE = """위 [AI 피드백]에서, [지원자 답변]에 없는 내용을 지원자에 대한 사실처럼 말한 문장을 모두 찾아라.

종류
- "사실처럼": 지원자가 하지 않은 말을 했다고 하거나, 답변에 없는 경력·활동·수치·결과를 지원자의 것으로 말함.
  답변 텍스트만으로는 알 수 없는 목소리·표정·시선·태도를 직접 본 것처럼 말한 것도 포함.
- "가상 예시": 고쳐 말하는 방법을 보여주려고 가상의 수치·사례를 예시로 든 것 ("예를 들어 '...'처럼 말하세요" 등).

세지 않는 것: 답변 내용에 대한 해석·평가, 면접관이 어떻게 받아들일지에 대한 의견, 일반적인 조언,
답변에 있는 내용을 다른 말로 바꿔 쓴 것. 확실하지 않으면 세지 않는다.

quote에는 AI 피드백의 문장을 한 글자도 바꾸지 말고 그대로 옮겨 적는다 (20자 이상).
출력 형식: {"items": [{"quote": "AI 피드백 원문 그대로", "type": "사실처럼|가상 예시", "why": "답변의 어디와 다른지 한 문장"}]}
없으면 {"items": []}"""


def norm(s: str) -> str:
    return re.sub(r'[\s"\'“”‘’]+', '', s or '')


def quote_in(quote: str, text: str) -> bool:
    """판정 LLM이 옮겨 적은 문장이 실제 피드백에 있는지 (공백·따옴표 무시, 말줄임 허용)"""
    q = norm(quote).strip('.…')
    t = norm(text)
    if len(q) < 10:
        return False
    if q in t:
        return True
    # 말줄임(…, ...)으로 이어 붙인 경우 조각마다 확인
    parts = [p for p in re.split(r'…|\.\.\.', q) if len(p) >= 10]
    return bool(parts) and all(p in t for p in parts)


def main():
    from dotenv import dotenv_values
    import anthropic

    client = anthropic.Anthropic(api_key=dotenv_values(BASE.parent / 'sk-dashboard' / 'backend' / '.env').get('ANTHROPIC_API_KEY', ''))
    usage = {'input': 0, 'output': 0, 'calls': 0}

    def ask(prompt, max_tokens=1500):
        # 인용문 속 따옴표 때문에 JSON이 깨지는 경우가 있어, 깨지면 다시 묻는다 (최대 4번)
        for _ in range(4):
            try:
                r = client.messages.create(model=JUDGE_MODEL, max_tokens=max_tokens, system=SYSTEM,
                                           messages=[{'role': 'user', 'content': prompt}])
            except Exception as e:
                print('   재시도:', str(e)[:60])
                continue
            with lock:
                usage['input'] += r.usage.input_tokens; usage['output'] += r.usage.output_tokens; usage['calls'] += 1
            text = ''.join(b.text for b in r.content if b.type == 'text')
            m = re.search(r'\{.*\}', text, re.S)
            try:
                return json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                print('   JSON 깨짐, 다시 묻기')
        return None

    gold = json.load(open(FULL / 'gold.json', encoding='utf-8'))['items']
    gen = json.load(open(FULL / 'generation.json', encoding='utf-8'))
    keys = [k for k in gold if all(gen.get(k, {}).get(v) for v in VERSIONS)]
    print(f'대상 {len(keys)}쌍 · 답변 최대 {max(len(gen[k]["interview"]) for k in keys)}자 (자르지 않음)')
    path = FULL / 'judge_unsupported.json'
    res = json.load(open(path, encoding='utf-8')) if path.exists() else {}

    def judge_item(k):
        g = gen[k]
        with lock:
            r = res.setdefault(k, {})
        for ver in VERSIONS:
            runs = [x for x in r.get(ver, []) if x is not None]
            while len(runs) < 2:
                out = ask(f'[지원자 답변]\n{g["interview"]}\n\n[AI 피드백]\n{g[ver]}\n\n{RULE}')
                if out is None:
                    break
                for it in out.get('items', []):
                    it['quote_found'] = quote_in(it.get('quote', ''), g[ver])
                runs.append(out)
            r[ver] = runs
        with lock:
            json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        cnt = lambda v: sum(it['type'] == '사실처럼' and it['quote_found'] for it in r[v][0].get('items', [])) if r[v] else '-'
        print(f'  {k:16s} 사실처럼 ' + ' | '.join(f'{v} {cnt(v)}' for v in VERSIONS))

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(judge_item, keys))

    # ── 요약 ──
    done = [k for k in keys if all(len(res.get(k, {}).get(v, [])) == 2 for v in VERSIONS)]
    for k in done:   # 인용문 확인은 현재 quote_in 기준으로 다시 계산
        for ver in VERSIONS:
            for run in res[k][ver]:
                for it in run.get('items', []):
                    it['quote_found'] = quote_in(it.get('quote', ''), gen[k][ver])
    json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    has = lambda run, t: any(it.get('type') == t and it.get('quote_found') for it in run.get('items', []))
    summ = {'judge_model': JUDGE_MODEL, 'pairs': len(done), 'versions': {}}
    for ver in VERSIONS:
        runs = [res[k][ver] for k in done]
        all_items = [it for rr in runs for run in rr for it in run.get('items', [])]
        summ['versions'][ver] = {
            'fabricated_pairs': f'{sum(has(run, "사실처럼") for rr in runs for run in rr) / 2:.1f}/{len(done)}',
            'hypothetical_example_pairs': f'{sum(has(run, "가상 예시") for rr in runs for run in rr) / 2:.1f}/{len(done)}',
            'two_runs_same_fabricated': f'{sum(has(rr[0], "사실처럼") == has(rr[1], "사실처럼") for rr in runs)}/{len(done)}',
            'quotes_not_found_dropped': f'{sum(not it.get("quote_found") for it in all_items)}/{len(all_items)}',
        }
    old = json.load(open(FULL / 'judge_summary.json', encoding='utf-8'))['versions']
    summ['old_unsupported_judge_full'] = {v: old[v]['unsupported'] for v in VERSIONS}
    summ['usage'] = usage
    summ['cost_usd_est'] = round(usage['input'] / 1e6 * PRICE_IN + usage['output'] / 1e6 * PRICE_OUT, 3)
    json.dump(summ, open(FULL / 'judge_unsupported_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n[지어냄 재판정 요약]', json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    main()
