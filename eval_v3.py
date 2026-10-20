"""
eval_v3.py — eval_v2와 같은 평가 6건에 v3(질의 정제 + 비슷한 답변 예시) 피드백을 생성해 generation_outputs.json에 추가

v1·v2와 같은 모델(gemini-3.5-flash)·같은 프롬프트(prompts_v2)를 쓰고, 검색 질의와 예시만 바꾼다.
예시 풀은 eval_v2와 같다 (평가 대상과 같은 영상의 쌍은 제외).
API 키: 무료 한도 때문에 sk-dashboard/backend/.env 의 GEMINI_API_KEY 사용
"""
import json, time

from eval_v2 import BASE, GEN_MODEL, OUT, load_pairs


def main():
    from dotenv import dotenv_values
    from google import genai
    from google.genai import types
    from rag_v2 import HybridRetriever
    from prompts_v2 import system_prompt
    from pipeline_v3 import ExampleSelector, rewrite_query

    client = genai.Client(api_key=dotenv_values(BASE.parent / 'sk-dashboard' / 'backend' / '.env').get('GEMINI_API_KEY', ''))
    pairs = load_pairs()
    test = pairs[::5][:6]
    test_videos = {p['video'] for p in test}
    selector = ExampleSelector([p for p in pairs if p['video'] not in test_videos])
    rag = HybridRetriever()

    path = OUT / 'generation_outputs.json'
    gen = json.load(open(path, encoding='utf-8'))
    for p in test:
        g = gen[p['id']]
        if g.get('v3'):
            continue
        for k in range(5):
            try:
                q = g.get('v3_query') or rewrite_query(client, GEN_MODEL, p['interview'])
                g['v3_query'] = q
                time.sleep(5)
                hits = rag.search(q, 4)
                shots = selector.select(p['interview'], 5)
                c = []
                for s in shots:
                    c += [types.Content(role='user', parts=[types.Part.from_text(text=f'Q.\n{s["interview"]}')]),
                          types.Content(role='model', parts=[types.Part.from_text(text=f'A.\n{s["feedback"]}')])]
                ctx = '\n\n'.join(f'[참고{i+1}] {h["text"]}' for i, h in enumerate(hits))
                c.append(types.Content(role='user', parts=[types.Part.from_text(
                    text=f'Q.\n{p["interview"]}\n\n--- 참고 자료 (면접왕 이형 강의) ---\n{ctx}')]))
                r = client.models.generate_content(model=GEN_MODEL, contents=c, config=types.GenerateContentConfig(
                    system_instruction=system_prompt(), temperature=0.4, max_output_tokens=4096))
                if r.text:
                    g['v3'] = r.text.strip().removeprefix('A.').strip()
                    g['v3_sources'] = [h['source'] for h in hits]
                    g['v3_shots'] = [s['id'] for s in shots]
                    break
            except Exception as e:
                if '429' in str(e):   # 하루 무료 한도 소진 — 재시도해도 횟수만 쓰므로 바로 멈춘다
                    print('   무료 한도 소진, 중단:', str(e)[:80])
                    json.dump(gen, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                    return
                print('   재시도:', str(e)[:80]); time.sleep(60)
        print(f"  {p['id']:14s} 질의: {g.get('v3_query', '')[:50]} | v3 {len(g.get('v3', ''))}자")
        json.dump(gen, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        time.sleep(5)


if __name__ == '__main__':
    main()
