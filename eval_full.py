"""
eval_full.py — feedback_data 31쌍 전체로 v1 / v2 / v3 피드백 생성 (같은 영상의 쌍은 예시에서 제외)

- v1: 발표자료 프롬프트(SYSTEM_PROMPT_V1) + 기존 DB 임베딩 검색(답변 앞 200자) + 무작위 예시 5개
- v2: prompts_v2 + 강의 DB 하이브리드 검색(답변 앞 300자) + v1과 같은 무작위 예시 5개
- v3: prompts_v2 + 질의 정제 후 하이브리드 검색 + 비슷한 답변 예시 5개
모델은 셋 다 gemini-3.5-flash, 출력 한도 4,096토큰, 답변 앞 "A." 제거.
결과: eval_results_v2/full/generation.json
"""
import json, os, random, re, shutil, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval_v2 import BASE, GEN_MODEL, OUT, PROJ, load_pairs

FULL = OUT / 'full'
FULL.mkdir(exist_ok=True)
lock = threading.Lock()


def main():
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    import chromadb
    from chromadb.utils import embedding_functions
    from rag_v2 import EMBED_MODEL, HybridRetriever
    from prompts_v2 import SYSTEM_PROMPT_V1, system_prompt
    from pipeline_v3 import ExampleSelector, REWRITE_PROMPT

    load_dotenv(PROJ / '.env')
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY', ''))
    pairs = load_pairs()
    tmp = Path(tempfile.mkdtemp()) / 'chroma_db_v1'
    shutil.copytree(PROJ / 'chroma_db', tmp)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    v1db = chromadb.PersistentClient(path=str(tmp)).get_collection('interview_rag', embedding_function=ef)
    rag = HybridRetriever()
    selectors = {}   # 영상별로 예시 풀이 달라서 영상마다 하나씩
    path = FULL / 'generation.json'
    gen = json.load(open(path, encoding='utf-8')) if path.exists() else {}

    def call(system, contents, max_tokens=4096):
        for k in range(5):
            try:
                cfg = types.GenerateContentConfig(system_instruction=system, temperature=0.4, max_output_tokens=max_tokens) if system \
                    else types.GenerateContentConfig(max_output_tokens=max_tokens)
                r = client.models.generate_content(model=GEN_MODEL, contents=contents, config=cfg)
                if r.text:
                    return re.sub(r'^\s*A\.\s*', '', r.text.strip())
            except Exception as e:
                wait = 60 if '429' in str(e) else 20
                print('   재시도:', str(e)[:60]); time.sleep(wait)
        return ''

    def contents_for(answer, ctx, shots):
        c = []
        for s in shots:
            c += [types.Content(role='user', parts=[types.Part.from_text(text=f'Q.\n{s["interview"]}')]),
                  types.Content(role='model', parts=[types.Part.from_text(text=f'A.\n{s["feedback"]}')])]
        c.append(types.Content(role='user', parts=[types.Part.from_text(
            text=f'Q.\n{answer}\n\n--- 참고 자료 (면접왕 이형 강의) ---\n{ctx}')]))
        return c

    def run(idx_p):
        idx, p = idx_p
        pool = [q for q in pairs if q['video'] != p['video']]
        with lock:
            g = gen.setdefault(p['id'], {'interview': p['interview'], 'coach': p['feedback']})
            if p['video'] not in selectors:
                selectors[p['video']] = ExampleSelector(pool)
        rand_shots = random.Random(idx).sample(pool, 5)
        if not g.get('v1'):
            docs = v1db.query(query_texts=[p['interview'][:200]], n_results=4)['documents'][0]
            g['v1_ctx'] = docs
            g['v1'] = call(SYSTEM_PROMPT_V1, contents_for(p['interview'], '\n\n'.join(f'[참고{i+1}] {d}' for i, d in enumerate(docs)), rand_shots))
        if not g.get('v2'):
            hits = rag.search(p['interview'][:300], 4)
            g['v2_ctx'] = [h['text'] for h in hits]
            g['v2'] = call(system_prompt(), contents_for(p['interview'], '\n\n'.join(f'[참고{i+1}] {h["text"]}' for i, h in enumerate(hits)), rand_shots))
        if not g.get('v3'):
            g['v3_query'] = g.get('v3_query') or call(None, REWRITE_PROMPT + p['interview'][:1500], 2048).strip('"') or p['interview'][:300]
            hits = rag.search(g['v3_query'], 4)
            g['v3_ctx'] = [h['text'] for h in hits]
            shots = selectors[p['video']].select(p['interview'], 5)
            g['v3_shots'] = [s['id'] for s in shots]
            g['v3'] = call(system_prompt(), contents_for(p['interview'], '\n\n'.join(f'[참고{i+1}] {h["text"]}' for i, h in enumerate(hits)), shots))
        with lock:
            json.dump(gen, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            done = sum(1 for v in gen.values() if v.get('v1') and v.get('v2') and v.get('v3'))
        print(f"  {p['id']:16s} v1 {len(g.get('v1', ''))} · v2 {len(g.get('v2', ''))} · v3 {len(g.get('v3', ''))}자 | 완료 {done}/{len(pairs)}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(run, list(enumerate(pairs))))
    print('완료')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    main()
