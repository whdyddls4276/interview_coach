"""
eval_v4.py — v4 = v3의 검색 질의·근거·예시 + v1의 프롬프트(발표자료 slide19 규칙)

31쌍 판정 결과 v3는 검색 근거 관련도가 가장 높았지만(47%) 코치 핵심 지적 재현율은 가장 낮았다(45%, v1 55%).
v2 프롬프트의 "6~10문장으로 짧게"·"잘한 건 잘했다고"(판단 쏠림 보정)가 원인인지 보려고
v3가 실제로 쓴 입력(v3_query·v3_ctx·v3_shots)을 그대로 두고 프롬프트만 v1으로 바꾼다.
  v4 vs v3 = 프롬프트 효과 / v4 vs v1 = 검색·예시 효과
결과: eval_results_v2/full/generation.json 에 v4 추가
"""
import json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor

from eval_v2 import GEN_MODEL, OUT, PROJ, load_pairs

FULL = OUT / 'full'
lock = threading.Lock()


def main():
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from prompts_v2 import SYSTEM_PROMPT_V1

    load_dotenv(PROJ / '.env')
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY', ''))
    pairs = {p['id']: p for p in load_pairs()}
    path = FULL / 'generation.json'
    gen = json.load(open(path, encoding='utf-8'))

    def run(k):
        g = gen[k]
        if g.get('v4'):
            return
        c = []
        for sid in g['v3_shots']:
            s = pairs[sid]
            c += [types.Content(role='user', parts=[types.Part.from_text(text=f'Q.\n{s["interview"]}')]),
                  types.Content(role='model', parts=[types.Part.from_text(text=f'A.\n{s["feedback"]}')])]
        ctx = '\n\n'.join(f'[참고{i+1}] {d}' for i, d in enumerate(g['v3_ctx']))
        c.append(types.Content(role='user', parts=[types.Part.from_text(
            text=f'Q.\n{g["interview"]}\n\n--- 참고 자료 (면접왕 이형 강의) ---\n{ctx}')]))
        for t in range(5):
            try:
                r = client.models.generate_content(model=GEN_MODEL, contents=c, config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_V1, temperature=0.4, max_output_tokens=4096))
                if r.text:
                    g['v4'] = re.sub(r'^\s*A\.\s*', '', r.text.strip())
                    g['v4_ctx'] = g['v3_ctx']
                    break
            except Exception as e:
                print('   재시도:', str(e)[:60]); time.sleep(60 if '429' in str(e) else 20)
        with lock:
            json.dump(gen, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            done = sum(1 for v in gen.values() if v.get('v4'))
        print(f'  {k:16s} v4 {len(g.get("v4", ""))}자 | 완료 {done}/{len(gen)}')

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(run, list(gen)))
    print('완료')


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    main()
